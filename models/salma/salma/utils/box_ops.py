###############################################################################
# 
#   This file was imported from the DETR codebase.
#   https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
# 
#   We only imported a minimal amount of tools to improve readability.
#   Modified for SALMA: unused components removed.
#
###############################################################################

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""

import torch
from torchvision.ops.boxes import box_area


OVERLAP_X_MIN = 840
OVERLAP_X_MAX = 1080

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    # While other conversion is safe, this one can lead to deegenerate boxes and need clamping.
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        print(f"{boxes1 = }, {boxes2 = }")
        
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def boxes_overlap(box_a: torch.Tensor, box_b: torch.Tensor) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = torch.max(ax1, bx1)
    inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2)
    inter_y2 = torch.min(ay2, by2)

    return bool((inter_x2 > inter_x1) & (inter_y2 > inter_y1))

def box_fully_in_overlap_strip(bbox: torch.Tensor) -> bool:
    x1, _, x2, _ = bbox
    return bool((x1 >= OVERLAP_X_MIN) & (x2 <= OVERLAP_X_MAX))

def box_intersects_overlap_strip(bbox: torch.Tensor) -> bool:
    x1, _, x2, _ = bbox
    return bool(~((x2 <= OVERLAP_X_MIN) | (x1 >= OVERLAP_X_MAX)))


def clip_box_to_overlap(bbox: torch.Tensor):
    x1, y1, x2, y2 = bbox
    cx1 = torch.clamp(x1, min=OVERLAP_X_MIN)
    cx2 = torch.clamp(x2, max=OVERLAP_X_MAX)

    if cx2 <= cx1:
        return None

    return torch.stack([cx1, y1, cx2, y2])


def hungarian_match_detections(
        left_dets: list,
        right_dets: list,
        matcher,
    ):
    l_boxes, l_idx = [], []
    r_boxes, r_idx = [], []

    # Process left detections
    for i, det in enumerate(left_dets):
        bbox = torch.tensor(det.bbox, dtype=torch.float32)
        clipped = clip_box_to_overlap(bbox)
        if clipped is not None:
            l_boxes.append(box_xyxy_to_cxcywh(clipped))
            l_idx.append(i)

    # Process right detections
    for j, det in enumerate(right_dets):
        bbox = torch.tensor(det.bbox, dtype=torch.float32)
        clipped = clip_box_to_overlap(bbox)
        if clipped is not None:
            r_boxes.append(box_xyxy_to_cxcywh(clipped))
            r_idx.append(j)

    # No valid overlap candidates
    if not l_boxes or not r_boxes:
        return []

    outputs = {
        "pred_boxes": torch.stack(l_boxes).unsqueeze(0),  # [1, N, 4]
    }

    targets = [{
        "boxes": torch.stack(r_boxes),  # [M, 4]
    }]

    indices = matcher(outputs, targets)[0]

    # Map back to original indices
    matched_pairs = [
        (l_idx[i], r_idx[j])
        for i, j in zip(indices[0].tolist(), indices[1].tolist())
    ]

    return matched_pairs
