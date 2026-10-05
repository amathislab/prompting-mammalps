"""
transforms.py

This module provides various image and video transformation utilities for preprocessing 
and augmenting data in video classification tasks.

Based on BEiT, timm, DINO, DeiT, and InternVideo code bases:
- https://github.com/microsoft/unilm/tree/master/beit
- https://github.com/rwightman/pytorch-image-models/tree/master/timm
- https://github.com/facebookresearch/deit
- https://github.com/facebookresearch/dino
- https://github.com/OpenGVLab/InternVideo

Modified for SALMA: unused components removed.
"""

import math
import numbers
import random

import numpy as np
import torch
import torchvision
from salma.utils.misc import InferenceMode

from torchvision.transforms import ColorJitter

from PIL import Image


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return (out_images, label)


class GroupHorizontalCropWithBoxes(object):
    def __init__(self, size, mode=InferenceMode.CENTER):
        self.input_size = size if not isinstance(size, int) else [size, size]
        assert mode in {InferenceMode.CENTER, InferenceMode.LEFT, InferenceMode.RIGHT}
        self.mode = mode

    def __call__(self, img_lab_box_tuple):
        img_group, label, sequence_boxes = img_lab_box_tuple
        im_w, im_h = img_group[0].size

        crop_size = min(im_w, im_h)
        offset_h = (im_h - crop_size) // 2

        if self.mode == InferenceMode.LEFT:
            offset_w = 0
        elif self.mode == InferenceMode.RIGHT:
            offset_w = im_w - crop_size
        else:  
            offset_w = (im_w - crop_size) // 2

        # --- crop and resize images ---
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_size, offset_h + crop_size))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), Image.BILINEAR)
            for img in crop_img_group
        ]

        # --- scale factors for boxes ---
        scale_x = self.input_size[0] / crop_size
        scale_y = self.input_size[1] / crop_size

        boxes_new_all = []
        for frame_boxes in sequence_boxes:
            if not frame_boxes:
                boxes_new_all.append([])
                continue

            frame_new = []
            for (x1, y1, x2, y2) in frame_boxes:
                # Adjust to crop and scale
                x1n = (x1 - offset_w) * scale_x
                y1n = (y1 - offset_h) * scale_y
                x2n = (x2 - offset_w) * scale_x
                y2n = (y2 - offset_h) * scale_y

                # Clip to bounds
                x1n = round(max(0, min(self.input_size[0], x1n)), 1)
                y1n = round(max(0, min(self.input_size[1], y1n)), 1)
                x2n = round(max(0, min(self.input_size[0], x2n)), 1)
                y2n = round(max(0, min(self.input_size[1], y2n)), 1)

                if x2n > x1n and y2n > y1n:
                    frame_new.append([x1n, y1n, x2n, y2n])

            boxes_new_all.append(frame_new)

        return ret_img_group, label, boxes_new_all

class GroupRandomHorizontalFlipWithBoxes:
    """
    Horizontally flips frames AND bounding boxes with probability p.
    """
    def __init__(self, p=0.5, selective_flip=False):
        self.p = p

    def __call__(self, data_tuple):
        frames, label, boxes = data_tuple

        if random.random() >= self.p:
            return frames, label, boxes

        # All frames have same size -> pick first
        W = frames[0].width

        # Flip images
        flipped_frames = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frames]

        # Flip boxes: (x1,y1,x2,y2) -> (W-x2, y1, W-x1, y2)
        flipped_boxes = []

        for frame_boxes in boxes:           # each frame has N boxes
            frame_flipped = []
            for (x1, y1, x2, y2) in frame_boxes:
                new_x1 = W - x2
                new_x2 = W - x1
                frame_flipped.append([new_x1, y1, new_x2, y2])
            flipped_boxes.append(frame_flipped)

        return (flipped_frames, label, flipped_boxes)

class GroupColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data_tuple):
        frames, label, boxes = data_tuple
        jittered = [self.jitter(img) for img in frames]
        return jittered, label, boxes


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_tuple):
        tensor, label = tensor_tuple
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))
        
        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return (tensor,label)

    
class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

class GroupMultiScaleCropWithBoxes(GroupMultiScaleCrop):
    """
    Extension of GroupMultiScaleCrop transform that handles bounding boxes in a
    similar manner. It is required to perform the exact same geometrical modifications
    for the bounding boxes to adapt their coordinates according to augmentation.
    """

    def __call__(self, img_lab_box_tuple):
        # Exact same behavior as parent function but tuple contains boxes.
        img_group, label, sequence_boxes = img_lab_box_tuple
        
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]

        # Scaling factors
        scale_x = self.input_size[0] / crop_w
        scale_y = self.input_size[1] / crop_h

        boxes_new_all = []
        for frame_boxes in sequence_boxes:
            # Empty boxes need to be explicitely stored.
            if not frame_boxes:
                boxes_new_all.append([])
                continue

            frame_new = []
            for (x1, y1, x2, y2) in frame_boxes:
                # Scaling coordinates
                x1n = (x1 - offset_w) * scale_x
                y1n = (y1 - offset_h) * scale_y
                x2n = (x2 - offset_w) * scale_x
                y2n = (y2 - offset_h) * scale_y

                # Clipping to image bounds
                x1n = round(max(0, min(self.input_size[0], x1n)), 1)
                y1n = round(max(0, min(self.input_size[1], y1n)), 1)
                x2n = round(max(0, min(self.input_size[0], x2n)), 1)
                y2n = round(max(0, min(self.input_size[1], y2n)), 1)

                # Only keep boxes overlapping the new frame.
                if x2n > x1n and y2n > y1n:
                    frame_new.append([x1n, y1n, x2n, y2n])

            boxes_new_all.append(frame_new)

        return ret_img_group, label, boxes_new_all
    
class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        if img_group[0].mode == 'L':
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple
        
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return (img.float().div(255.) if self.div else img.float(), label)
