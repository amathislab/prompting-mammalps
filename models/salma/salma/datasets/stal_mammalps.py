from __future__ import annotations

from pathlib import Path

import logging

logger = logging.getLogger(__name__)

from PIL import Image
import json
import math
import torch
import numpy as np
import csv
import random
from torch.utils.data import DistributedSampler
from typing import Iterator, Optional
from einops import rearrange

from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset
from torchvision import transforms
from salma.utils.misc import Mode, InferenceMode, CSVFields, VideoFormat
from salma.utils.box_ops import box_xyxy_to_cxcywh
from salma.datasets.video_reader import video_loader
from salma.transforms.transforms import (GroupNormalize, 
                                            Stack, 
                                            ToTorchFormatTensor, 
                                            GroupMultiScaleCropWithBoxes, 
                                            GroupHorizontalCropWithBoxes,
                                            GroupColorJitter,
                                            GroupRandomHorizontalFlipWithBoxes)

BBox = list[float]                  # A bounding box is a list of coordinates [x1, y1, x2, y2].
FrameBoxes = list[BBox]             # For a single frame, there can be multiple bounding boxes.
SequenceBoxes = list[FrameBoxes]    # A video contains several frames.

AnimalLabel = list[str, torch.Tensor]     # A label is a dictionary of one hot encoded species, actions and activity for a specific individual
FrameLabel = list[AnimalLabel]            # A frame has several animals, each having a label.
SequenceLabel = list[FrameLabel]          # A sequence has several frames.


class MammalpsForSTAL(Dataset):
    """
    Dataset used for Spatio-Temporal Action Tracking. It constructs video
    sequences and their corresponding bounding boxes, species, actions and activity 
    on a per-frame basis.

    It performs augmentation based on the mode (train/test/val). During training, 
    it captures random portions of the image with different zoom levels. During 
    inference, it simply takes the largest centered video compatible with image size. 
    For example, with img_size = (224,224), it selects the center square of size 
    1080x1080 from the original 1920x1080 image and resize it to 224x224.

    Attributes:
        dense_annot_root (str | Path): Absolute path to the dense annotations folder.

        video_root (str | Path): Absolute path to the "raw_videos" folder.

        label_mapping (dict): Dictionary of activities, actions and species indices.
            (See metadata/label_mapping.json)

        transform (AugmentationForSTALWithBoxes): Transform used to format and augment 
            sequences.

        mode (Mode): The mode used (Mode.TRAIN | Mode.VALID | Mode.TEST).

        img_size (tuple(int,int)): Size of the out image.

        chunk_len (int): Number of frames to sample the sequence from.

        num_frames (int): Number of frames of the out sequence.

        fps (float): Camera FPS.

        video_ext (VideoFormat): Format of video file 
            (VideoFormat.MP4 is the only one supported).
        
        threads (int): Number of threads used in the video loader.
        
        split_csv_path (str): Absolute path to the csv split file.

    """
    def __init__(self,
                 dense_annot_root: str | Path,
                 split_csv_path: str | Path,
                 video_root: str | Path,
                 label_mapping: dict[str, dict[str, int]],
                 transform: AugmentationForSTALWithBoxes,
                 mode: Mode = Mode.TRAIN,
                 img_size: tuple[int, int] = (224, 224),
                 num_frames: int = 16,
                 video_ext: VideoFormat = VideoFormat.MP4,
                 threads: int = 1,
                 tubelet_size: int = 2,
                 test_cache_size: int = 16,
                 jitter: bool = False,
                 temporal_stride: int = 1,
                 min_stride: Optional[int] = None,
                 max_stride: Optional[int] = None
                 ):
        

        
        super().__init__()  

        self.transform = transform
        self.video_ext = video_ext
        self.video_root = video_root
        self.num_frames = num_frames
        self.img_size = img_size
        self.threads = threads
        self.label_mapping = label_mapping
        self.mode = mode
        self.tubelet_size = tubelet_size
        self.test_cache_size = test_cache_size
        self.temporal_stride = temporal_stride
        self.min_stride = min_stride # Only for training
        self.max_stride = max_stride # Only for training
        self.jitter = jitter
    
        self.videos_info = []

        self.dense_annot_root = Path(dense_annot_root)
        assert self.dense_annot_root.is_dir(), f"{self.dense_annot_root} is not a valid directory."
                
        # Reading info from current split raw videos.
        with open(split_csv_path) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip header

            for row in csv_reader:
                video_data = self._parse_raw_video_info_line(row)
                self.videos_info.append(video_data)
        
        self.videos_info = self.videos_info
        logger.info(f"[{mode}] Loaded {len(self.videos_info)} videos from {split_csv_path}")

        if self.mode == Mode.TEST:
            self._video_cache = {}

    def __len__(self):
        #if self.mode == Mode.TEST:
        #    return int(np.sum([(v["duration"] * v["fps"]) for v in self.videos_info]) // self.num_frames)
        #else:
        #    return len(self.videos_info)
        return len(self.videos_info)
    
    def _get_video_from_cache(
        self,
        vid_rel_path: str,
        duration: float,
        fps: float,
        temporal_stride: int = 1,
        max_cache_size: int = 64,
        jitter: bool = False,
        ):

        """
        Sequentially consume frames from a cached video chunk.
        Cache behaves as a FIFO queue sampled with temporal_stride.
        Returns None and throws warning once the video is exhausted.
        """

        total_frames = int(math.ceil(duration * fps))

        cache = self._video_cache.setdefault(
            vid_rel_path,
            {"frames": [], "frame_ids": [], "cursor": 0},
        )

        # Refill cache if needed
        if len(cache["frames"]) < self.num_frames and cache["cursor"] < total_frames:
            chunk_start = cache["cursor"]
            cache_window = min(temporal_stride * max_cache_size, (total_frames - chunk_start))
            chunk_end = min(chunk_start + cache_window, total_frames)
            num_frames = cache_window // temporal_stride

            start_sec = chunk_start / fps
            end_sec = chunk_end / fps

            frames, frame_ids = video_loader(
                root=self.video_root,
                vid=vid_rel_path,
                ext=self.video_ext,
                start_second=start_sec,
                end_second=end_sec,
                fps=fps,
                chunk_len=-1,
                num_frames=num_frames,
                threads=self.threads,
                fast_rrc=False,
                rrc_params=None,
                jitter=jitter,
            )

            dedup = dict()

            for fid, frame in zip(frame_ids, frames):
                dedup.setdefault(fid, frame)

            cache["frame_ids"].extend(dedup.keys())
            cache["frames"].extend(dedup.values())

            cache["cursor"] = chunk_end

        # End of video and nothing left to serve
        if not cache["frames"]:
            logger.debug(f"Video '{vid_rel_path}' exhausted.")
            return None

        # Pop next window
        frames_out = cache["frames"][:self.num_frames]
        frame_ids_out = cache["frame_ids"][:self.num_frames]

        # Duplicating last frame to have sequence of correct length.
        while len(frames_out) < self.num_frames:
            frames_out.append(frames_out[-1])
            frame_ids_out.append(frame_ids_out[-1])


        cache["frames"] = cache["frames"][self.num_frames:]
        cache["frame_ids"] = cache["frame_ids"][self.num_frames:]

        return {
            "frames": frames_out,
            "frame_ids": frame_ids_out,
        }


    def __getitem__(self, index, cropping_mode: InferenceMode | None = None ):
        video_info = self.videos_info[index] 

        vid_rel_path =  video_info["vid_rel_path"]
        annot_file_path = Path(self.dense_annot_root) / video_info["annot_rel_path"]
        duration = video_info["duration"]
        fps = video_info["fps"]

        clip_path = Path(self.video_root) / Path(vid_rel_path).with_suffix(f".{self.video_ext}")
        assert clip_path.is_file(), f"{clip_path} is not a valid file."
        
        if self.mode == Mode.TRAIN and self.min_stride is not None and self.max_stride is not None:
            temporal_stride = random.randint(self.min_stride, self.max_stride)
        else:
            temporal_stride = self.temporal_stride

        chunk_len = temporal_stride * self.num_frames
        
        if self.mode == Mode.TEST:
            # Sequentially sample frames.
            video_data = self._get_video_from_cache(
                vid_rel_path = vid_rel_path,
                duration = duration,
                fps = fps,
                temporal_stride = temporal_stride,
                jitter = self.jitter,
                max_cache_size=self.test_cache_size
                )

            if video_data is None: # When exhausted video, return None as failsafe.
                return None

            frames = video_data["frames"]
            frame_ids = video_data["frame_ids"]

            if cropping_mode in [InferenceMode.LEFT, InferenceMode.RIGHT]:
                self.transform.spatial_transform   = GroupHorizontalCropWithBoxes(self.img_size, cropping_mode)
                
                
        else:
            # Take a random sequence from the video.
            frames, frame_ids = video_loader(
                root=self.video_root, 
                vid=vid_rel_path, 
                ext=self.video_ext,
                start_second = 0.,
                end_second = duration,
                chunk_len=chunk_len, 
                fps=fps,
                num_frames=self.num_frames,
                threads=self.threads,
                jitter=self.jitter,
                )
        
        video_meta = {"file_id": Path(vid_rel_path).stem,
                "file_path": vid_rel_path,
                "frame_ids": frame_ids,
                "num_frames": duration * fps,
                "duration_s": duration,
                "fps": fps,
            }  
                       
        frames = [Image.fromarray(frame).convert('RGB') for frame in frames]
        labels, bboxes, track_ids, weather_oh = self._get_labels_from_frames(annot_file_path = annot_file_path, frame_ids = frame_ids)
        process_data, bboxes = self.transform((frames, None, bboxes))
        process_data = rearrange(process_data, '(t c) h w -> c t h w', t=self.num_frames, c=3) # More elegant than initial VideoMAE implem...
        
        H, W = self.img_size 
        valid_bboxes = []
        valid_labels = []
        valid_track_ids = []

        for f,(frame_bboxes, frame_labels) in enumerate(zip(bboxes, labels)):
            # Convert to tensors
            if isinstance(frame_bboxes, list):
                frame_bboxes = (
                    torch.tensor(frame_bboxes, dtype=torch.float32)
                    if len(frame_bboxes) > 0
                    else torch.empty((0, 4))
                )

            # Handling empty boxes and labels.
            if frame_bboxes.numel() == 0:
                valid_bboxes.append(torch.empty((0, 4), device=torch.device("cpu")))
                valid_labels.append([])
                valid_track_ids.append([])
                continue

            # Clamp and filter boxes within the cropped region
            x1, y1, x2, y2 = frame_bboxes.unbind(dim=1)
            box_w = x2 - x1
            box_h = y2 - y1
            valid_mask = (box_w > 1) & (box_h > 1)

            filtered_boxes = frame_bboxes[valid_mask]
            if filtered_boxes.numel() > 0:
                filtered_boxes = torch.stack([
                    filtered_boxes[:, 0] / W,
                    filtered_boxes[:, 1] / H,
                    filtered_boxes[:, 2] / W,
                    filtered_boxes[:, 3] / H
                ], dim=1)
                
            filtered_labels = [lab for lab, keep in zip(frame_labels, valid_mask.tolist()) if keep]
            filtered_tracks = [tid for tid, keep in zip(track_ids[f], valid_mask.tolist()) if keep]

            valid_bboxes.append(filtered_boxes)
            valid_labels.append(filtered_labels)
            valid_track_ids.append(filtered_tracks)

        
        T = len(valid_bboxes)
        T_tubelet = T // self.tubelet_size
        targets = []

       
        # Merging frame classes and boxes over tracklet size.
        for t_tubelet in range(T_tubelet):
            frame_indices = list(range(
                self.tubelet_size * t_tubelet,
                self.tubelet_size * (t_tubelet + 1)
            ))
            frame_dicts = []
            for f in frame_indices:
                frame_boxes = valid_bboxes[f]
                frame_labels = valid_labels[f]
                frame_tracks = valid_track_ids[f]

                d = {
                    tid: (box, lab)
                    for box, lab, tid in 
                    zip(frame_boxes, frame_labels, frame_tracks)
                }
                frame_dicts.append(d)

            all_tids = set().union(*[d.keys() for d in frame_dicts])

            merged_boxes = []
            merged_species = []
            merged_activities = []
            merged_actions = []
            merged_dage = []
            merged_dsex = []

            # Merging similar track ids over adjacent frames.
            for tid in sorted(all_tids): 

                tid_boxes = []
                tid_labels = []

                for d in frame_dicts:
                    if tid in d:
                        box, lab = d[tid]
                        tid_boxes.append(box)
                        tid_labels.append(lab)
                
                # We take the tracklet box as the union of the frame boxes
                tid_boxes = torch.stack(tid_boxes)
                union_box = torch.tensor([
                    torch.min(tid_boxes[:, 0]),
                    torch.min(tid_boxes[:, 1]),
                    torch.max(tid_boxes[:, 2]),
                    torch.max(tid_boxes[:, 3]),
                ])
                # convert xyxy -> cxcywh
                union_box = box_xyxy_to_cxcywh(union_box)

                # We take the first frame as the label of the over frames
                attrs = tid_labels[0]

               
                merged_boxes.append(union_box)
                merged_species.append(attrs["species"])
                merged_activities.append(attrs["activity"])
                merged_actions.append(attrs["action"])
                merged_dage.append(attrs["dage"])
                merged_dsex.append(attrs["dsex"])
            
            if merged_boxes:
                merged_boxes = torch.stack(merged_boxes)
                merged_species = torch.stack(merged_species).type(torch.float16)
                merged_activities = torch.stack(merged_activities).type(torch.float16)
                merged_actions = torch.stack(merged_actions).type(torch.float16)
                merged_dage = torch.stack(merged_dage).type(torch.float16)
                merged_dsex = torch.stack(merged_dsex).type(torch.float16)
            else:
                merged_boxes = torch.zeros((0, 4)) 
                merged_species = torch.zeros((0,len(self.label_mapping["species"])), dtype=torch.float16)
                merged_activities = torch.zeros((0,len(self.label_mapping["activities"])), dtype=torch.float16)
                merged_actions = torch.zeros((0, len(self.label_mapping["actions"])), dtype=torch.float16)    
                merged_dage = torch.zeros((0,len(self.label_mapping["deer_ages"])), dtype=torch.float16)   
                merged_dsex = torch.zeros((0,len(self.label_mapping["deer_adult_sexes"])), dtype=torch.float16)

            targets.append({
                "boxes": merged_boxes,
                "species": merged_species,
                "activities": merged_activities,
                "actions": merged_actions,
                "dages": merged_dage,
                "dsexes": merged_dsex,
                "weather": weather_oh,
                "frames": torch.tensor(frame_indices),
                "track_ids": torch.tensor(sorted(all_tids)),
            })

        data = {
            "video": process_data,
            "targets": targets,
            "meta" : video_meta,
        }

        return data
    
    def _parse_raw_video_info_line(self, 
                                   row: list[str], 
                                   ) -> dict:
        """
        Parse a CSV file Stal MammAlps format and returns
        relevant fields in a structured dict.
        """

        EXPECTED_FIELDS = [f.name.lower() for f in CSVFields]

        expected_len = len(EXPECTED_FIELDS)
        if len(row) != expected_len:
            raise ValueError(
                f"Invalid row format: expected {expected_len} fields ({EXPECTED_FIELDS}), got {len(row)} -> {row}"
            )

        try:
            video_file_path = row[CSVFields.VIDEO_FILE_PATH]
            annot_file_path = row[CSVFields.ANNOT_FILE_PATH]
            duration = float(row[CSVFields.DURATION])
            fps = float(row[CSVFields.FPS])
            vid_rel_path = str(Path(video_file_path).with_suffix(""))
            sampling_weight = float(row[CSVFields.SAMPLING_WEIGHT])

            video_data = {
                "vid_rel_path": vid_rel_path,
                "annot_rel_path": annot_file_path,
                "duration": duration,
                "fps": fps,
                "sampling_weight": sampling_weight
            }

            return video_data

        except (ValueError, IndexError, TypeError) as e:
            raise ValueError(f"Error parsing row: {row} -> {e}") from e

    def _get_labels_from_frames(self,
                                annot_file_path: str | Path,
                                frame_ids: list[int]
                                ) -> tuple[SequenceLabel, SequenceBoxes, list, torch.tensor]:
        try:
            with open(annot_file_path, "r") as f:
                frames_data = json.load(f)

            all_frames = frames_data.get("frames", [])
            weather_data = frames_data["info"]["attributes"].get("weather", "unknown")
            labels = []
            bboxes = []
            track_ids = []

            num_species = len(self.label_mapping["species"])
            num_actions = len(self.label_mapping["actions"])
            num_activities = len(self.label_mapping["activities"])
            num_dage = len(self.label_mapping["deer_ages"])
            num_dsex = len(self.label_mapping["deer_adult_sexes"])
            num_weather = len(self.label_mapping["weather"])

            weather_oh = torch.zeros(num_weather)
            weather_idx = self.label_mapping["weather"].get(weather_data, -1)
            if weather_idx >= 0:
                weather_oh[weather_idx] = 1

            for fid in frame_ids:
                if fid >= len(all_frames):
                    labels.append([])
                    bboxes.append([])
                    continue

                detections = all_frames[fid].get("detections", [])
                frame_labels = []
                frame_bboxes = []
                frame_track_ids = []

                for det in detections:
                    track_id = det.get("track_id", None)
                    bbox = det.get("bbox", None)
                    attrs = det.get("attributes", {})

                    # --- extract categorical names
                    species_name = attrs.get("Species", "unknown")
                    action_name = attrs.get("Action", "unknown")
                    action2_name = attrs.get("Action2", "none") # Second class should be none.
                    activity_name = attrs.get("Activity", "unknown")
                    deer_age_name = attrs.get("Deer_age", "unknown")
                    adult_deer_sex_name = attrs.get("Deer_adult_sex", "unknown")

                    # map to index
                    species_idx = self.label_mapping["species"].get(species_name, -1)
                    action_idx = self.label_mapping["actions"].get(action_name, -1)
                    action2_idx = self.label_mapping["actions"].get(action2_name, -1)
                    activity_idx = self.label_mapping["activities"].get(activity_name, -1)
                    dage_idx = self.label_mapping["deer_ages"].get(deer_age_name, -1)
                    dsex_idx = self.label_mapping["deer_adult_sexes"].get(adult_deer_sex_name, -1)

                    # convert to one-hot tensors
                    species_oh = torch.zeros(num_species)
                    if species_idx >= 0:
                        species_oh[species_idx] = 1

                    action_oh = torch.zeros(num_actions)
                    if action_idx >= 0:
                        action_oh[action_idx] = 1
                    if action2_idx >= 0:
                        action_oh[action2_idx] = 1

                    activity_oh = torch.zeros(num_activities)
                    if activity_idx >= 0:
                        activity_oh[activity_idx] = 1
                    
                    dage_oh = torch.zeros(num_dage)
                    if dage_idx >= 0:
                        dage_oh[dage_idx] = 1
                    
                    dsex_oh = torch.zeros(num_dsex)
                    if dsex_idx >= 0:
                        dsex_oh[dsex_idx] = 1
                    
                    frame_labels.append({
                        "species": species_oh,
                        "action": action_oh,
                        "activity": activity_oh,
                        "dage": dage_oh,
                        "dsex": dsex_oh,
                    })

                    frame_bboxes.append(bbox)
                    frame_track_ids.append(track_id)

                labels.append(frame_labels)
                bboxes.append(frame_bboxes)
                track_ids.append(frame_track_ids)

            return (labels, bboxes, track_ids, weather_oh)

        except OSError as oserr:
            logger.error(f"OS error occurred trying to open {annot_file_path}: {oserr}")
            return [], [], [], []

    
class AugmentationForSTALWithBoxes(object):
    def __init__(
            self,
            img_size=(224, 224),
            mode=Mode.TRAIN,
            multiscale_scales=[1, .875, .75, .66],
            hflip_p=0.5,
            color_jitter_params=(0.4, 0.4, 0.4, 0.1),
        ):

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std  = [0.229, 0.224, 0.225]
        self.mode = mode

        normalize = GroupNormalize(self.input_mean, self.input_std)

        if self.mode == Mode.TRAIN:
            b, c, s, h = color_jitter_params

            self.spatial_transform = transforms.Compose([
                GroupMultiScaleCropWithBoxes(img_size, multiscale_scales),
                GroupRandomHorizontalFlipWithBoxes(p=hflip_p),
            ])

            self.color_transform = GroupColorJitter(
                brightness=b, contrast=c, saturation=s, hue=h
            )

        else :
            self.spatial_transform   = GroupHorizontalCropWithBoxes(img_size, InferenceMode.CENTER) # Initialized with center zoom.
            self.color_transform = lambda data: data

        self.frame_transform = transforms.Compose([
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])


    def __call__(self, data_tuple):
        frames, label, boxes = data_tuple

        frames, label, boxes = self.spatial_transform((frames, label, boxes))
        frames, label, boxes = self.color_transform((frames, label, boxes))

        process_data, _ = self.frame_transform((frames, label))

        return process_data, boxes

    def __repr__(self):
        repr = "(AugmentationForSTALWithBoxes,\n"
        repr += f"  frame_transform = {self.frame_transform},\n"
        repr += f"  spatial_transform = {self.spatial_transform},\n"
        repr += f"  color_transform = {self.color_transform}\n"
        repr += ")"
        return repr
    
def build_stal_mammalps(csv_path: str | Path, 
                        dense_annot_root: str | Path, 
                        label_mapping: dict[str, dict[str, int]], 
                        video_root: str |Path, 
                        mode: Mode = Mode.TRAIN, 
                        img_size: tuple[int, int] = (224, 224),
                        num_frames: int = 16,
                        video_ext: str = VideoFormat.MP4,
                        threads: int = 1,
                        test_cache_size: int = 64,
                        jitter: bool = False,
                        temporal_stride: int = 1,
                        min_stride: Optional[int] = None, 
                        max_stride: Optional[int] = None,
                        ):
    
    transform = AugmentationForSTALWithBoxes(img_size=img_size, mode=mode)

    split_csv_path = (Path(csv_path) / mode).with_suffix(".csv")

    dataset = MammalpsForSTAL(
        dense_annot_root=dense_annot_root,
        video_root=video_root,
        mode=mode,
        label_mapping=label_mapping,
        split_csv_path=split_csv_path,
        transform=transform,
        img_size=img_size,
        num_frames=num_frames,
        video_ext=video_ext,
        threads=threads,
        test_cache_size=test_cache_size,
        temporal_stride=temporal_stride,
        min_stride=min_stride,
        max_stride=max_stride,
        jitter=jitter
        )
    
    logger.info("Data Aug = %s" % str(transform))
    return dataset
    

class MammalpsSTALDistributedBalancingSampler(DistributedSampler[int]):
    """
    A distributed sampler for balancing classes in MammAlps.

    Assigns class weights for every sample as inverse of class frequency
    Then select a fixed number of samples with replacement according to the sample weights

    Replacement is usually set to true to draw multiple times the samples from the minority
    class (up sampling) and on average different samples from the majority class (down sampling)

    Attributes:
        labels (list): Labels corresponding to the samples.
        num_samples_per_epoch (int): Number of samples to draw for each epoch.
        replacement (bool): Whether to sample with replacement.
    """

    def __init__(self, dataset: MammalpsForSTAL, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0, drop_last: bool = False, replacement: bool = True,
                 smoothing_factor: float = .001):
        """
        Initialize the distributed balancing sampler.

        Args:
            dataset (MammalpsDataset): The dataset to sample from.
            num_replicas (int, optional): Number of replicas in distributed training.
            rank (int, optional): Rank of the current process.
            seed (int): Random seed.
            drop_last (bool): Whether to drop the last incomplete batch.
            replacement (bool): Whether to sample with replacement.
            smoothing_factor (float): Smoothing factor for class weights.
        """
        
        super(MammalpsSTALDistributedBalancingSampler, self).__init__(dataset, num_replicas, rank, shuffle=True, seed=seed, drop_last=drop_last)

        self.replacement = replacement
        self.sample_weights = np.array([v.get("sampling_weight", 1.0) + smoothing_factor for v in dataset.videos_info]) 
        self.sample_weights /= np.max(self.sample_weights)
        self.sample_weights = torch.Tensor(self.sample_weights)

    def __iter__(self) -> Iterator[int]:
        """
        Build a random iterator with upsampling strategy.

        Returns:
            Iterator[int]: Iterator over sample indices.
        """

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.sample_weights, len(self.dataset), self.replacement, generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
        

def stal_collate_fn(batch):
        """
        Collate function for Mammalps samples with fixed-shape videos and
        variable-length target lists. Videos are stacked into a batch tensor,
        while per-sample 'targets' (lists of box/label dictionaries) are kept
        unmodified in a list because their sizes differ across samples.
        """

        videos = default_collate([d["video"] for d in batch])
        targets = [d["targets"] for d in batch]
        meta = [d["meta"] for d in batch]

        return {"video": videos, "targets": targets, "meta": meta}