###############################################################################
# 
#   This file was imported from the DETR codebase.
#   See https://github.com/facebookresearch/detr/blob/main/util/misc.py
# 
#   We only imported a minimal amount of tools to improve readability.
#   Modified for SALMA: unused components removed.
#
###############################################################################

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

import torch.distributed as dist
from enum import StrEnum, IntEnum, auto
import json
import logging

def load_label_mapping(
        label_mapping_path: str,
        logger,
) -> dict:
    
    label_mapping = {}
    
    try:
        with open(label_mapping_path, "r") as f:
            label_mapping = json.load(f)
    except Exception as e:
        logger.error(f"Could not load label mapping config file {label_mapping_path}")
        raise(e)

    return label_mapping

def reverse_label_mapping(
    label_mapping: dict[str, dict[str, int]],
) -> dict[str, dict[int, str]]:
    return {
        group: {v: k for k, v in mapping.items()}
        for group, mapping in label_mapping.items()
    }


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
    def filter(self, record):
        record.rank = self.rank
        return True

def get_logger_with_rank(rank):
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[Rank %(rank)s] %(asctime)s %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(RankFilter(rank))
    return logger

class CSVFields(IntEnum):
    VIDEO_FILE_PATH = 0
    ANNOT_FILE_PATH = 1
    DURATION = 2
    FPS = 3
    SAMPLING_WEIGHT = 4

class Mode(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

class VideoFormat(StrEnum):
    MP4 = auto()
    MOV = auto()
    AVI = auto()
    MKV = auto()

class InferenceMode(StrEnum):
    LEFT = auto()
    RIGHT = auto()
    CENTER = auto()
    MULTI_VIEW = auto()
