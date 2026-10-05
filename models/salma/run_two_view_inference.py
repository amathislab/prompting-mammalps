import json
from pathlib import Path
import torch
from timm import create_model
import numpy as np
import argparse
from copy import deepcopy
import logging

from salma.postproc.merging_preds import merge_left_right, pad_right_bounding_boxes
import salma.model.salma.Salma  # Needed to register model for timm.create_model.
from salma.utils.misc import (
                                Mode, 
                                VideoFormat, 
                                InferenceMode, 
                                load_label_mapping, 
                                reverse_label_mapping, 
                                get_logger_with_rank,
)

from salma.datasets.stal_mammalps import build_stal_mammalps, stal_collate_fn
from salma.utils.box_ops import box_cxcywh_to_xyxy
from salma.datasets.mammalps_dataloader import build_test_loader
from salma.model.utils import init_distributed_mode, is_dist_avail_and_initialized
from salma.postproc.annot_utils import (
                                            ObservationStore,
                                            add_obs,
                                            aggregate_video_attrs,
                                            aggregate_tracks,
                                            materialize_frames,
                                            make_empty_video_annot,
                                        )
from salma.postproc.constants import ATTRIBUTE_SPECS
from salma.postproc.data_enums import AnnotAttr, AnnotLevel, LabelMappingKey
from salma.postproc.datastructures import Detection, VideoAnnot
from salma.model.salma.head_enum import SalmaHead

from collections import OrderedDict

import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = None

def parse_args():
    p = argparse.ArgumentParser(description="Overlapping Two-View Inference on MammAlps.")

    p.add_argument("--experiment_name", type=str, default="default_exp")
    p.add_argument("--output_dir", type=str, default="./output")

    p.add_argument("--model_name", type=str, default="salma_base_patch16_224")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--label_mapping_path", type=str, required=True)
    p.add_argument("--dense_annot_path", type=str, required=True)
    p.add_argument("--video_root_path", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--temporal_stride", type=int, default=1)
    p.add_argument("--frame_stack_width", type=int, default=2)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    p.add_argument("--video_resolution", type=int, nargs=2, default=(1920, 1080))
    p.add_argument("--num_queries", type=int, default=20)
    p.add_argument("--test_cache_size", type=int, default=64, help="Number of video frames to load in memory at once")

    p.add_argument("--use_lora", action="store_true", default=False, help='Enable LoRA.')
    p.add_argument("--jitter", action="store_true", default=False, help='Turns on temporal jittering during training.')


    return p.parse_args()


def fill_missing_frames(video: VideoAnnot) -> VideoAnnot:
    if not video.frames:
        return video

    frames = sorted(video.frames, key=lambda f: f.frame_id)
    existing_ids = [f.frame_id for f in frames]
    id_map = {f.frame_id: f for f in frames}

    start = 0
    end = existing_ids[-1]

    filled = []
    j = 0 

    for fid in range(start, end + 1):
        if j < len(existing_ids) and fid == existing_ids[j]:
            filled.append(id_map[fid])
            if j < len(existing_ids) - 1:
                j += 1
        else:
            src = id_map[existing_ids[j]]
            new_frame = deepcopy(src)
            new_frame.frame_id = fid
            filled.append(new_frame)

    video.frames = filled
    return video

def aggregate_video(
        vid: int,
        view: InferenceMode,
        data: dict[str, object],
        id2label_mapping: dict[str, dict[int, str]],
        width: int,
        height: int,
        video_resolution: tuple[int, int] = (1920, 1080),
    ) -> VideoAnnot:
    """
    Aggregates a single video's predictions and returns a fully
    materialized + hole-filled VideoAnnot.
    """

    store = data["store"]
    video_annot: VideoAnnot = data["annot"]

    # --- VIDEO attributes ---
    video_annot.info.attributes = aggregate_video_attrs(store, id2label_mapping)

    # --- TRACK attributes ---
    track_attrs = aggregate_tracks(store, id2label_mapping)

    # --- FRAME materialization ---
    frames_out = materialize_frames(
        store,
        track_attrs,
        id2label_mapping,
        width,
        height,
    )

    video_annot.frames = frames_out

    # --- Fill missing frames ---
    video_annot = fill_missing_frames(video_annot)

    # --- Right view coordinate adjustment ---
    if view == InferenceMode.RIGHT:
        pad_right_bounding_boxes(video_annot, *video_resolution)

    return video_annot

def write_video_annot(
        video_annot: VideoAnnot,
        output_dir: str | Path,
    ):
    """
    Writes a fully processed VideoAnnot to disk.
    """

    output_json_path = (Path(output_dir) / video_annot.info.file_id).with_suffix(".json")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(video_annot.to_dict(), f, indent=2)

def infer_single_view(model, dataloader, id2label_mapping, output_dir, img_size, frame_stack_width, view, args, device="cuda"):
    model.eval()
    model.set_mode(Mode.TEST)
    model.to(device)

    video_predictions_dict = {}
    width, height = img_size

    last_vid_per_slot = {}

    with torch.no_grad():
        for data in dataloader:
            batch = data["batch"]
            batch_meta = data["meta"]
            slot_ids = batch_meta["slot_id"]
            video_ids = batch_meta["video_index"]

            batch_videos = batch["video"].to(device, non_blocking=True)

            # Inference pass
            out = model(batch_videos, slot_ids, video_ids) # Why giving the video_ids ?
            
            # Get predictions
            pred_is_animal = torch.sigmoid(out[SalmaHead.IS_ANIMAL]).squeeze(-1)  # (B, T, Q)
            pred_classes = {}
            conf_classes = {}
            for attr, spec in ATTRIBUTE_SPECS.items():
                if spec.head in out:
                    probs = torch.softmax(out[spec.head], dim=-1)
                    if attr == AnnotAttr.ACTION2:
                        pred_classes[attr] = torch.topk(probs, k=2, dim=-1).indices[:, :, :, 1]
                        conf_classes[attr] = torch.topk(probs, k=2, dim=-1).values[:, :, :, 1]
                    else:
                        pred_classes[attr] = torch.max(probs, dim=-1).indices
                        conf_classes[attr] = torch.max(probs, dim=-1).values

            pred_boxes = out[SalmaHead.BOXES]  # (B, T, Q)

            # Get valid animal objects
            pred_masks = (pred_is_animal) >= 0.5

            # For each video present in the batch, we collect the predictions
            for v, vid in enumerate(video_ids):
                vid = int(vid)
                vid_meta = batch["meta"][v]
                slot_id = slot_ids[v]

                if slot_id in last_vid_per_slot:
                    previous_vid = last_vid_per_slot[slot_id]

                    if vid != previous_vid:
                        # Flush previous video
                        if previous_vid in video_predictions_dict:
                            video_annot = aggregate_video(
                                previous_vid,
                                view,
                                video_predictions_dict[previous_vid],
                                id2label_mapping,
                                width,
                                height,
                                args.video_resolution,
                            )
                            
                            write_video_annot(video_annot, output_dir)
                            del video_predictions_dict[previous_vid]

                last_vid_per_slot[slot_id] = vid

                if vid not in video_predictions_dict.keys():
                    video_predictions_dict[vid] = {
                        "annot": make_empty_video_annot(vid_meta),
                        "track_id_map": {},
                        "store": ObservationStore(),
                    }

                store = video_predictions_dict[vid]["store"]

                # VIDEO level observations
                for attr, spec in ATTRIBUTE_SPECS.items():
                    if spec.level == AnnotLevel.VIDEO:
                        value = int(pred_classes[attr][v].item())
                        add_obs(store, attr, value)

                # FRAME and TRACK level observations
                frame_ids = np.array(vid_meta["frame_ids"])

                # Skipping repeating index at the end of videos
                _, unique_indices = np.unique(frame_ids, return_index=True)
                frame_ids_unique = frame_ids[np.sort(unique_indices)]

                for t, frame_id in enumerate(frame_ids_unique):
                    tracklet_index = t // frame_stack_width
                    valid_queries_idx = torch.nonzero(pred_masks[v, tracklet_index], as_tuple=True)[0]

                    store.register_frame(int(frame_id)) # Register frame,
    
                    for qi in valid_queries_idx:
                        qi_int = int(qi.item())
                        if qi_int not in video_predictions_dict[vid]["track_id_map"].keys():
                            video_predictions_dict[vid]["track_id_map"][qi_int] = len(video_predictions_dict[vid]["track_id_map"])
                        track_id = video_predictions_dict[vid]["track_id_map"][qi_int]

                        # Box & confidence
                        box = pred_boxes[v, tracklet_index, qi_int]
                        box_xyxy = box_cxcywh_to_xyxy(box).clamp(0,1) # Avoid degenerate boxes

                        box_xyxy[[0, 2]] *= width
                        box_xyxy[[1, 3]] *= height
                        x1, y1, x2, y2 = map(int, box_xyxy.tolist())
                        conf = float(pred_is_animal[v, tracklet_index, qi_int].item())

                        # Frame detection attributes
                        attrs = {}
                        for attr in [AnnotAttr.ACTIVITY, AnnotAttr.ACTION, AnnotAttr.ACTION2]:
                            val = pred_classes[attr][v, tracklet_index, qi_int]
                            conf_cls = conf_classes[attr][v, tracklet_index, qi_int]
                            if attr == AnnotAttr.ACTION2:
                                val = int(val.item()) if conf_cls>=0.1 else -1
                            else:
                                val = int(val.item())
                            attrs[attr] = val

                        # Register FRAME detection
                        det = Detection(track_id=track_id, bbox=[x1, y1, x2, y2],
                                             conf=conf, attributes=attrs)
                        
                        #print(f"{vid} / {frame_id.item() = }: {track_id = }, {conf = }")
                        

                        add_obs(store, AnnotAttr.ACTIVITY, det, frame_id=int(frame_id))  # just push det

                        # TRACK level observations
                        for attr in [AnnotAttr.SPECIES, AnnotAttr.DEER_AGE, AnnotAttr.DEER_SEX]:
                            val = int(pred_classes[attr][v, tracklet_index, qi_int].item())
                            add_obs(store, attr, val, track_id=track_id)
                

        
        for vid in list(video_predictions_dict.keys()):
            print(f"Writing {vid = } to {output_dir}.")

            video_annot = aggregate_video(
                vid,
                view,
                video_predictions_dict[vid],
                id2label_mapping,
                width,
                height,
                args.video_resolution,
            )

            write_video_annot(video_annot, output_dir)

            del video_predictions_dict[vid]

        torch.cuda.empty_cache()

def run_single_view(
    view_name: str,
    device: str,
    args: argparse.Namespace,
    label_mapping: dict[str, dict[str, int]],
    id2label_mapping: dict[str, dict[int, str]],
    video_resolution: tuple[int],
    logger = None
):
    torch.cuda.set_device(device) if "cuda" in device else None

    nb_species = len(label_mapping[LabelMappingKey.SPECIES])
    nb_actions = len(label_mapping[LabelMappingKey.ACTIONS])
    nb_activities = len(label_mapping[LabelMappingKey.ACTIVITY])
    nb_dage = len(label_mapping[LabelMappingKey.DEER_AGE])
    nb_dsex = len(label_mapping[LabelMappingKey.DEER_SEX])
    nb_weather = len(label_mapping[LabelMappingKey.WEATHER])


    # MODEL
    model = create_model(
        args.model_name,
        pretrained=True,
        encoder_weights=None,
        num_classes=None,
        use_lora=args.use_lora,
        num_queries=args.num_queries,
        num_species=nb_species,
        num_activities=nb_activities,
        num_actions=nb_actions, 
        num_dage=nb_dage,
        num_dsex=nb_dsex,
        num_weather=nb_weather
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Strip "module." prefix
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v

    # Rename legacy key
    if "object_queries.weight" in new_sd:
        new_sd["first_object_queries.weight"] = new_sd.pop("object_queries.weight")

    # Rename object queries -> animal queries
    if "first_object_queries.weight" in new_sd:
        if "first_animal_queries.weight" not in new_sd:
            new_sd["first_animal_queries.weight"] = new_sd["first_object_queries.weight"]
        del new_sd["first_object_queries.weight"]

    load_result = model.load_state_dict(new_sd, strict=False)

    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys

    print("\nCHECKPOINT LOADING REPORT")

    if missing_keys:
        print("\n[Missing keys] (expected by model, NOT found in checkpoint):")
        for k in missing_keys:
            print(f"  - {k}")
    else:
        print("\n[Missing keys] None")

    if unexpected_keys:
        print("\n[Unexpected keys] (found in checkpoint, NOT used by model):")
        for k in unexpected_keys:
            print(f"  - {k}")
    else:
        print("\n[Unexpected keys] None")

    print("\n================================\n")

    # DATALOADER
    dataset = build_stal_mammalps(csv_path = args.csv_path, 
                                dense_annot_root = args.dense_annot_path, 
                                label_mapping = label_mapping, 
                                video_root = args.video_root_path, 
                                mode = Mode.TEST,
                                img_size = args.img_size,
                                num_frames = args.num_frames,
                                video_ext = VideoFormat.MP4,
                                test_cache_size=args.test_cache_size,
                                temporal_stride = args.temporal_stride,
                                jitter = args.jitter,
                            )

    dataloader = build_test_loader(
        dataset = dataset,
        batch_size=args.batch_size,
        collate_fn = stal_collate_fn,
        cropping_mode = view_name,
        logger = logger
    )

    out_dir = Path(args.output_dir) / view_name
    out_dir.mkdir(parents=True, exist_ok=True)

    square_video_resolution = (min(video_resolution), min(video_resolution))
    infer_single_view(model=model, 
                dataloader=dataloader, 
                id2label_mapping = id2label_mapping,
                output_dir=out_dir,
                img_size= square_video_resolution,
                frame_stack_width=args.frame_stack_width,
                device=device,
                view=view_name,
                args=args
            )


def main():
    logger = get_logger_with_rank(0)
    logger.setLevel(logging.INFO)
    args = parse_args()
    init_distributed_mode(args)
    
    label_mapping = load_label_mapping(args.label_mapping_path, logger=logger)
    id2label_mapping = reverse_label_mapping(label_mapping)


    rank = args.rank
    world_size = args.world_size

    gpu = args.gpu if args.distributed else None
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    views = [InferenceMode.LEFT, InferenceMode.RIGHT]

    for view in views:
        logger.info(f"Rank {rank}: running {view} inference on GPU {gpu}")
        run_single_view(
            view_name=view,
            device=device,
            args=args,
            label_mapping=label_mapping,
            id2label_mapping=id2label_mapping,
            video_resolution = args.video_resolution,
            logger=logger
        )

    # make sure all JSON files are written
    # But one process might finish much earlier than the others. 
    # Use with caution. Potential timeout
    if is_dist_avail_and_initialized():
        dist.barrier()

    if rank == 0:
        out_path = Path(args.output_dir)
        merged_folder = out_path / InferenceMode.MULTI_VIEW
        merged_folder.mkdir(exist_ok = True)
        
        merge_left_right(left_folder=out_path / InferenceMode.LEFT, 
                        right_folder=out_path / InferenceMode.RIGHT, 
                        output_folder=merged_folder, 
                        out_img_size=args.video_resolution,
                        logger = logger,
                        )

    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
