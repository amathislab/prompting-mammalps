"""
video_reader.py

This module provides utilities for reading and processing video data, including frame extraction, 
and segmentation map handling.

Based on Avion code base:
- https://github.com/zhaoyue-zephyrus/AVION

Modified for SALMA: unused components removed.
"""

import os.path as osp
import numpy as np
import decord

def get_frame_ids(start_frame, end_frame, num_frames=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_frames + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_frames
        shift = (np.random.rand(num_frames) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()

def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader

def video_loader(root, vid, ext, start_second, end_second,
                 chunk_len=300, fps=30, num_frames=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0'
    assert osp.exists(osp.join(root, '{}.{}'.format(vid, ext))), '{}.{} does not exist!'.format(vid, ext)

    vr = get_video_reader(
        osp.join(root, '{}.{}'.format(vid, ext)),
        num_threads=threads,
        fast_rrc=fast_rrc, rrc_params=rrc_params,
        fast_rcc=fast_rcc, rcc_params=rcc_params,
    )

    end_second = min(end_second, len(vr) / fps)
    total_duration_f = min((end_second - start_second) * fps, len(vr))

    if chunk_len == -1 or total_duration_f <= chunk_len:
        # We sample frames between provided start and end times
        start_frame = int(np.round(start_second * fps))
        end_frame = int(min(start_frame + total_duration_f, len(vr)))
    
    else:
        # We select a random chunk within the provided timestamps
        start_frame = np.random.randint(int(np.round(start_second * fps)), np.ceil(end_second * fps - chunk_len + 1))
        end_frame = int(min(start_frame + chunk_len, len(vr)))

    frame_ids = get_frame_ids(start_frame, end_frame, num_frames=num_frames, jitter=jitter)

    # load frames
    try:
        assert max(frame_ids) < len(vr)
        frames = vr.get_batch(frame_ids).asnumpy()
    except Exception as error:
        print(error)
        #frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
        frames = np.array([])
        frame_ids = np.array([])

    # return torch.from_numpy(frames.astype(np.float32))
    return frames, frame_ids
