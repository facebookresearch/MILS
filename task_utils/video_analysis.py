# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from dataclasses import dataclass
import io
import math
import random
import traceback
from enum import Enum
from typing import Any, BinaryIO, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import glob
import torch
import torchvision
import torchvision.transforms.functional as F


import numpy as np

import torch

import torchvision.transforms as transforms

from PIL import Image
import decord  # type: ignore
decord.bridge.set_bridge("torch")


class DecoderType(str, Enum):
    DECORD = "decord"
    TORCHVISION = "torchvision"



def _sample_frame_ids(
    video_length_range,
    sampling_fps,
    num_frames,
    actual_fps,
    num_frames_per_video,
    video_identifier=None,
):
    assert video_length_range[0] <= video_length_range[1]

    # Figure out num_frames_per_video of indices to extract with equal spacing
    if sampling_fps is not None:
        assert (
            num_frames_per_video is None
        ), "When sampling_fps specified, num_frames will be auto-determined based on video length"
        num_frames_per_video = int(num_frames * sampling_fps / actual_fps)
    elif num_frames_per_video is not None:
        assert (
            sampling_fps is None
        ), "When num_frames_per_video is specified, sampling_fps is auto-determined based on video length"
    else:
        # If neither of num_frames_per_video or sampling_fps is specified,
        # return all the frames. The chunker will then figure out what
        # chunks to extract features from.
        num_frames_per_video = num_frames

    if num_frames < num_frames_per_video:
        print(
            f"number of frames ({num_frames}) in the video is less "
            f"than number of frames per video ({num_frames_per_video}). "
            f"Video identifier: {video_identifier}. "
            f"Will repeat the frames to get {num_frames_per_video}."
        )

    # minimum and maximum consecutive frames to sample from
    min_consecutive_frames = int(min(video_length_range[0] * actual_fps, num_frames))
    max_consecutive_frames = int(min(video_length_range[1] * actual_fps, num_frames))

    # sample the number of consecutive frames we will use
    consecutive_frames = random.randint(min_consecutive_frames, max_consecutive_frames)

    # sample the starting frame index
    start_frame_index = random.randint(0, num_frames - consecutive_frames)

    # our sampling range is [start_frame_index, start_frame_index + consecutive_frames)
    # uniformly sample frames in this range
    frame_indices = np.linspace(
        start_frame_index,
        start_frame_index + consecutive_frames - 1,
        num_frames_per_video,
        dtype=int,
    )
    return frame_indices

def read_frames(
    fp: BinaryIO,
    num_frames_per_video: int,
    video_length_range: Sequence,
    video_frames_stack_style: Tuple[int] = (1, 1),
    sampling_fps: Optional[float] = None,
    video_identifier: Optional[str] = None,  # only used for logging
    decoder: DecoderType = DecoderType.DECORD,
):
    if decoder == DecoderType.DECORD:
        av_reader = decord.VideoReader(uri=fp)
        actual_fps = av_reader.get_avg_fps()
        num_frames = len(av_reader)
    else:
        av_reader = torchvision.io.VideoReader(src=fp.read())
        actual_fps = float(av_reader.container.streams.video[0].average_rate)
        # duration = float(
        #     av_reader.container.streams.video[0].duration
        #     * av_reader.container.streams.video[0].time_base
        # )
        # num_frames = int(duration * actual_fps)

        # FIXME: this is inefficient if we want to decode only short clips
        # within a video
        # NOTE: we also want to make sure we don't have way too many frames in memory
        # so we use a helper function which keeps dynamically subsampling frames
        # to a maximum size
        torchvision_max_frames_buffer = num_frames_per_video * 2
        frames = subsample_iterator(av_reader, torchvision_max_frames_buffer)
        assert len(frames) <= torchvision_max_frames_buffer
        frames = [frame["data"] for frame in frames]
        num_frames = len(frames)

    frame_indices = _sample_frame_ids(
        video_length_range,
        sampling_fps,
        num_frames,
        actual_fps,
        num_frames_per_video,
        video_identifier=video_identifier,
    )

    if decoder == DecoderType.DECORD:
        # Extract the frames and apply transformations
        frames = av_reader.get_batch(frame_indices)
        av_reader.seek(0)
        # NF x H x W x C -> NF x C x H x W
        frames = frames.permute(0, 3, 1, 2)
    else:
        # in torchvision we are decoding the whole video
        # this can be made more efficient by only decoding the frames we need
        # but for now in our setup anyway we use the full videos

        # shape is already NF x C x H x W, but need to extract the right frame ids
        frames = torch.stack(frames)
        frames = frames[frame_indices]

    stacked_frames = stack_frames(frames, video_frames_stack_style)

    return stacked_frames


def stack_frames(frames, video_frames_stack_style):
    """
    Args
        frames (Nf x C x H x W)
        video_frames_stack_style (num_rows, num_cols)
    """
    num_rows, num_cols = video_frames_stack_style
    if num_rows == 1 and num_cols == 1:
        return frames
    # Stack the frames in the shape specified by video_frames_stack_style
    nf, c, h, w = frames.size()
    assert (
        nf % (num_rows * num_cols) == 0
    ), f"Stack shape mismatch: {nf}, {num_rows}, {num_cols}"
    stacked_frames = frames.unsqueeze(1).unsqueeze(1)
    stacked_frames = (
        stacked_frames.reshape(-1, num_rows, num_cols, c, h, w)
        .permute(0, 3, 1, 4, 2, 5)
        .flatten(2, 3)
        .flatten(3, 4)
    )
    return stacked_frames


def subsample_iterator(iterator: Iterator, max_size: int) -> List[Any]:
    """Subsample elements from an iterator to save CPU memory.

    This function will read data from an iterator and uniformly subsample elements
    if necessary.

    - If the length of the iterator is less than max_size, we will return the entire
        iterator.
    - Otherwise, the returned elements count will be in the range
        [max_size // 2, max_size). In this case, the samples are subsampled with
        a frequencey which is an exponent of 2, i.e. 2^k.

    See https://www.internalfb.com/intern/anp/view/?id=5241917 to see the returned
    outputs in various settings.

    Args:
        iterator: an iterator that yields data
        max_size: the maximum number of elements to hold in memory. should be divisible
            by 2.
    Returns:
        A list of data. The length of the list will be less than max_size.
    """
    assert max_size % 2 == 0
    # we use two buffers, so to make sure we stay within the max_size
    # we use a buffer size of half of that
    buffer_size = max_size // 2
    curr_freq = 1
    subsampled_data = []
    curr_data = []

    for idx, data in enumerate(iterator):
        if idx % curr_freq == 0:
            curr_data.append(data)
        if len(curr_data) == buffer_size:
            # we have reached the buffer_size, so we need to subsample
            was_subsampled_data_empty = len(subsampled_data) == 0
            subsampled_data += curr_data
            curr_data = []
            if not was_subsampled_data_empty:
                # we already had subsampled elements (which will always be the case
                # after the first round), so we need to subsample this list to
                # bring it back to buffer_size
                assert (
                    len(subsampled_data) == 2 * buffer_size
                ), f"{len(subsampled_data)}"
                subsampled_data = subsampled_data[::2]
                # now that we've subsampled frames at double the frequency, we should
                # only add new frames at double the frequency
                curr_freq *= 2
    return subsampled_data + curr_data


def convert_videos_to_images(input_path='/path/to/nextqa/videos', output_path='/path/to/nextqa/videos_8'):
    for path in glob.glob(os.path.join(input_path, '*')):
        for file in glob.glob(os.path.join(path, '*.mp4')):
            new_path = os.path.join(output_path, os.path.split(path)[-1])
            os.makedirs(new_path, exist_ok=True)
            with open(file, "rb") as fp:
                x = read_frames(
                    fp, 
                    8,
                    video_length_range=[float("inf"), float("inf")])
                for i, y in enumerate(x):
                    # TODO: very important for more than one frame: pad with zeros the *.png
                    F.to_pil_image(y).save(os.path.join(new_path, os.path.split(file)[-1].replace('.mp4', f'_{i}.png')))