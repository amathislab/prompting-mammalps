import json
import os
from collections import defaultdict
from dataclasses import replace

import torch
from salma.losses.HungarianMatcher import HungarianMatcher
from salma.utils.box_ops import ( boxes_overlap,
                                box_intersects_overlap_strip,
                                hungarian_match_detections,
                                box_iou,
                                box_fully_in_overlap_strip,
                                )

from salma.postproc.datastructures import VideoAnnot, FrameAnnot, AnnotAttr, Detection
from salma.utils.misc import InferenceMode

def collect_match_statistics(
        left_video: VideoAnnot,
        right_video: VideoAnnot,
        matcher,
    ) -> tuple[dict[tuple[int,int], int], dict[tuple[int,int], int]]:
    """
    Compute matching statistics between left and right tracks, 
    as well as frames where both tracks exist but are NOT candidates
    for merging (coexistence outside overlapping region).

    Returns:
        match_counts: dict[(left_id, right_id)] -> number of frames matched by Hungarian matcher
        coexist_nonmerge_counts: dict[(left_id, right_id)] -> frames where left/right both exist but are non-overlapping
    """

    match_counts = defaultdict(int)
    coexist_nonmerge_counts = defaultdict(int)

    # Index frames by frame_id
    lf = {f.frame_id: f for f in left_video.frames}
    rf = {f.frame_id: f for f in right_video.frames}

    frame_ids = sorted(set(lf) | set(rf))

    for fid in frame_ids:
        left_frame = lf.get(fid)
        right_frame = rf.get(fid)

        left_dets = left_frame.detections if left_frame else []
        right_dets = right_frame.detections if right_frame else []

        # Candidate detections in overlapping region (merging candidates)
        left_cand_idx = candidates_in_overlap(left_dets)
        right_cand_idx = candidates_in_overlap(right_dets)

        sub_left = [left_dets[i] for i in left_cand_idx]
        sub_right = [right_dets[j] for j in right_cand_idx]

        # Hungarian matching for this frame (standard)
        if sub_left and sub_right:
            matches = hungarian_match_detections(sub_left, sub_right, matcher)
            for li, ri in matches:
                l_id = sub_left[li].track_id
                r_id = sub_right[ri].track_id
                match_counts[(l_id, r_id)] += 1
    return match_counts

def resolve_identities(
    match_counts: dict[tuple[int, int], int],
    left_ids: set[int],
    right_ids: set[int],
    left_conf: dict[int, float],
    right_conf: dict[int, float],
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Merge left/right tracks into global IDs based on matching statistics.

    Args:
        match_counts: (left_id, right_id) -> number of matched frames
        left_ids: set of all left track IDs
        right_ids: set of all right track IDs
        left_conf: left track confidence sums
        right_conf: right track confidence sums

    Returns:
        left_to_global, right_to_global: mapping from left/right IDs to unified global IDs
    """

    left_to_global: dict[int, int] = {}
    right_to_global: dict[int, int] = {}
    global_id = 0

    counts = dict(match_counts)

    while counts:
        (l, r), _ = max(counts.items(), key=lambda x: x[1])

        # Dominant track decides which side's labels to keep
        if left_conf[l] >= right_conf[r]:
            dominant = InferenceMode.LEFT
        else:
            dominant = InferenceMode.RIGHT

        left_to_global[l] = global_id
        right_to_global[r] = global_id
        global_id += 1

        # Remove conflicting matches
        counts = {
            (ll, rr): c
            for (ll, rr), c in counts.items()
            if ll != l and rr != r
        }

    # Assign new global IDs to unmatched tracks
    for l in left_ids:
        if l not in left_to_global:
            left_to_global[l] = global_id
            global_id += 1

    for r in right_ids:
        if r not in right_to_global:
            right_to_global[r] = global_id
            global_id += 1

    return left_to_global, right_to_global


def candidates_in_overlap(dets: list[Detection]) -> list[int]:
    idx = []
    for i, d in enumerate(dets):
        bbox = torch.tensor(d.bbox)
        if box_intersects_overlap_strip(bbox):
            idx.append(i)
    return idx
    
def track_regularizer(
        video: VideoAnnot,
        track_conf_sum: dict[int, float],
        track_frame_count: dict[int, set[int]],
        iou_thresh: float = 0.9,
        track_overlap_thresh: float = 0.6,
        min_track_length: int = 20,
    ) -> VideoAnnot:

    overlap_count = defaultdict(int)
    coexist_count = defaultdict(int)

    for frame in video.frames:
        dets = frame.detections

        # Compare all detection pairs in the frame
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                d1, d2 = dets[i], dets[j]
                t1, t2 = d1.track_id, d2.track_id

                if t1 == t2:
                    continue

                # Species-aware suppression
                if d1.attributes.get(AnnotAttr.SPECIES) != d2.attributes.get(AnnotAttr.SPECIES):
                    continue

                key = tuple(sorted((t1, t2)))
                coexist_count[key] += 1

                b1 = torch.tensor(d1.bbox, dtype=torch.float32).unsqueeze(0)
                b2 = torch.tensor(d2.bbox, dtype=torch.float32).unsqueeze(0)
                iou_val, _ = box_iou(b1, b2)

                if iou_val.item() > iou_thresh:
                    overlap_count[key] += 1

    # Determine tracks to remove
    tracks_to_remove = set()

    for (t1, t2), co_cnt in coexist_count.items():
        if co_cnt == 0:
            continue
        ov = overlap_count.get((t1, t2), 0)
        if ov / co_cnt >= track_overlap_thresh:
            if track_conf_sum[t1] >= track_conf_sum[t2]:
                tracks_to_remove.add(t2)
            else:
                tracks_to_remove.add(t1)

    # Remove tracks shorter than min_track_length
    for tid, frames_seen in track_frame_count.items():
        if len(frames_seen) < min_track_length:
            tracks_to_remove.add(tid)

    # Filter out detections for removed tracks
    for frame in video.frames:
        frame.detections = [
            det for det in frame.detections
            if det.track_id not in tracks_to_remove
        ]

    return video



def merge_predictions(
        left_video: VideoAnnot,
        right_video: VideoAnnot,
        left_to_global: dict[int, int],
        right_to_global: dict[int, int],
        output_width: int,
        output_height: int,
        logger = None,
    ) -> VideoAnnot:

    # Preserve left video metadata
    merged_video = VideoAnnot(info=left_video.info)

    # Index frames by frame_id
    lf = {f.frame_id: f for f in left_video.frames}
    rf = {f.frame_id: f for f in right_video.frames}

    for fid in sorted(set(lf) | set(rf)):

        left_frame = lf.get(fid)
        right_frame = rf.get(fid)

        left_dets = left_frame.detections if left_frame else []
        right_dets = right_frame.detections if right_frame else []

        out_detections: list[Detection] = []
        used_r = set()

        # Merge left detections
        for l_det in left_dets:
            l_gid = left_to_global[l_det.track_id]
            l_box = torch.tensor(l_det.bbox, dtype=torch.float32)
            if box_fully_in_overlap_strip(l_box):
                if logger is not None:
                    logger.debug(f"Found box in the overlap strip: {l_box} for {merged_video.info.file_id}, {fid}")
                continue

            merged_flag = False

            for j, r_det in enumerate(right_dets):
                if j in used_r or right_to_global[r_det.track_id] != l_gid:
                    continue

                r_box = torch.tensor(r_det.bbox, dtype=torch.float32)
                if box_fully_in_overlap_strip(r_box):
                    if logger is not None:
                        logger.debug(f"Found box in the overlap strip: {r_box} for {merged_video.info.file_id}, {fid}")
                    continue

                if boxes_overlap(l_box, r_box):
                    # Keep left detection but assign global id
                    out_detections.append(
                        replace(l_det, track_id=l_gid)
                    )
                    used_r.add(j)
                    merged_flag = True
                    break

            if not merged_flag:
                out_detections.append(
                    replace(l_det, track_id=l_gid)
                )

        # Add unmatched right detections
        for j, r_det in enumerate(right_dets):
            r_box = torch.tensor(r_det.bbox, dtype=torch.float32)
            if box_fully_in_overlap_strip(r_box):
                if logger is not None:
                        logger.debug(f"Found box in the overlap strip: {r_box} for {merged_video.info.file_id}, {fid}")
                continue

            if j not in used_r:
                out_detections.append(
                    replace(r_det, track_id=right_to_global[r_det.track_id])
                )

        seen_tids = set()
        unique_out_dets: list[Detection] = []

        for det in out_detections:
            tid = det.track_id
            if not tid in seen_tids:
                seen_tids.add(tid)
                unique_out_dets.append(det)

        merged_video.frames.append(
            FrameAnnot(
                frame_id=fid,
                detections=unique_out_dets,
                width=output_width,
                height=output_height,
            )
        )

    return merged_video



def pad_right_bounding_boxes(
        video: VideoAnnot,
        output_height: int,
        output_width: int,
    ) -> None:
    """
    Pads bounding boxes in-place for RIGHT view VideoAnnot.
    """

    for frame in video.frames:
        pad_x = output_height - frame.height
        pad_y = output_width - frame.width

        if not frame.detections:
            continue

        for det in frame.detections:
            if not det.bbox or len(det.bbox) != 4:
                continue

            x1, y1, x2, y2 = det.bbox
            det.bbox = [
                int(x1 + pad_x),
                int(y1 + pad_y),
                int(x2 + pad_x),
                int(y2 + pad_y),
            ]

def merge_left_right(left_folder, right_folder, output_folder, out_img_size, logger = None):

    matcher = HungarianMatcher(cost_bbox=1, cost_giou=1, cost_class=0)

    left_files = [f for f in os.listdir(left_folder) if f.endswith(".json")]

    for filename in left_files:

        left_path = os.path.join(left_folder, filename)
        right_path = os.path.join(right_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(left_path, "r") as f:
            left_json = json.load(f)
            left_video_annot = VideoAnnot.from_dict(left_json)

        with open(right_path, "r") as f:
            right_json = json.load(f)
            right_video_annot = VideoAnnot.from_dict(right_json)
        
        left_conf, _ = left_video_annot.compute_track_statistics()
        right_conf, _ = right_video_annot.compute_track_statistics()

        left_ids = set(left_conf.keys())
        right_ids = set(right_conf.keys())


        # ---- hungarian matching ----
        match_counts = collect_match_statistics(
            left_video_annot,
            right_video_annot,
            matcher
        )

        left_to_global, right_to_global = resolve_identities(
            match_counts,
            left_ids,
            right_ids,
            left_conf,
            right_conf,
        )


        # ---- merge ----
        merged_video_annot = merge_predictions(
            left_video_annot,
            right_video_annot,
            left_to_global,
            right_to_global,
            *out_img_size,
            logger = logger
        )

        # ---- recompute global track stats after merge ----        
        merged_conf, merged_frames = merged_video_annot.compute_track_statistics()

        # ---- final regularization ----
        merged_video_annot = track_regularizer(
            merged_video_annot,
            merged_conf,
            merged_frames,
            iou_thresh=0.8,
            track_overlap_thresh=0.7,
            min_track_length=5,
        )

        with open(output_path, "w") as f:
            json.dump(merged_video_annot.to_dict(), f, indent=2)

        if logger is not None:
            logger.info(f"[DONE] {filename}\n")

