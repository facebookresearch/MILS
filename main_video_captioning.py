# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import transformers
import argparse
import tqdm
import random
import json
from open_clip import create_model_and_transforms, get_tokenizer
from optimization_utils import (
    Scorer as S,
    Generator as G,
    get_image_features,
)

from task_utils.video.utils import (
    _frame_from_video,
    get_clip,
    frames2tensor,
    get_vid_feat,
    get_text_feat_dict,
)
import torch
import numpy as np
import cv2

from paths import VIDEOC_MSRVTT_ANNOTATIONS, VIDEOC_MSRVTT_VIDEOS, OUTPUT_DIR


def get_v_text_features(model, tokenizer, lines, device, batch_size):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.


    model:
        CLIP-like model with `encode_text`

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    lines: list of str
        name of classes

    Returns
    -------

    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    text_feat_d = {}
    text_feat_d = get_text_feat_dict(lines, model, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in lines]
    text_feats_tensor = torch.cat(text_feats, 0)
    return text_feats_tensor


@torch.no_grad()
def get_video_features(clip, video_paths, device, batch_size):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.


    model:
        CLIP-like model with `encode_text`

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    Returns
    -------

    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    all_vid_feat = []
    clip = clip.to(device)
    for video_path in video_paths:
        video = cv2.VideoCapture(video_path)
        video.open(video_path)
        frames = [x for x in _frame_from_video(video)]
        frames_tensor = frames2tensor(frames, device=device)
        vid_feat = get_vid_feat(frames_tensor, clip)
        all_vid_feat.append(vid_feat)
    return torch.concatenate(all_vid_feat, axis=0)


def optimize_for_videos(
    args, text_pipeline, video_ids, text_prompt, clip_model, tokenizer, 
):
    loggers = {}
    save_locations = {}
    for video_id in video_ids:
        save_locations[f"{video_id}"] = os.path.join(args.output_dir, f"{video_id}")
        os.makedirs(save_locations[f"{video_id}"], exist_ok=True)
        loggers[f"{video_id}"] = open(
            os.path.join(save_locations[f"{video_id}"], "log.txt"), "w"
        )

    generator = G(
        text_pipeline,
        args.text_model,
        requested_number=args.requested_number,
        keep_previous=args.keep_previous,
        prompt=text_prompt,
        key=lambda x: -x[0],
        batch_size=args.batch_size,
        device=args.device,
        exploration=args.exploration,
    )
    video_paths = [
        os.path.join(args.videos_path, video_id + ".mp4")
        for video_id in video_ids
    ]
    target_features = (
        get_video_features(
            clip_model,
            video_paths,
            args.device,
            args.batch_size,
        )
        .detach()
        .cpu()
        .numpy()
    )

    def clip_scorer(sentences, target_feature):
        text_features = get_v_text_features(
            clip_model,
            tokenizer,
            sentences,
            args.device,
            args.batch_size,
        )
        return text_features.detach().cpu().numpy() @ target_feature

    scorers = {}
    for i, video_id in enumerate(video_ids):
        scorers[f"{video_id}"] = {
            "func": clip_scorer,
            "target_feature": target_features[i],
        }

    scorer = S(
        scorers, args.batch_size, key=lambda x: -x, keep_previous=args.keep_previous
    )
    ###
    # Initialize the pool
    ###
    with open(args.init_descriptions, "r") as w:
        init_sentences = [i.strip() for i in w.readlines()]
    lines_with_scores = {}
    initial_scores = {}
    for i, video_id in enumerate(video_ids):
        init_scores = scorer.score(f"{video_id}", init_sentences)
        lines_with_scores[f"{video_id}"] = [
            (s, l) for (s, l) in zip(init_scores, init_sentences)
        ]
        best_score = sorted(lines_with_scores[f"{video_id}"], key=lambda x: -x[0])[0]
        initial_scores[f"{video_id}"] = best_score
        mean_score = np.mean(init_scores)
        bs = best_score[1].strip()
        loggers[f"{video_id}"].write(f"{best_score[0]}\t{mean_score}\t{bs}\n")
    ###
    # Do the optimization:
    ###
    for it in range(args.iterations):
        torch.cuda.empty_cache()
        new_lines = generator(lines_with_scores)
        # new_lines is similar to lines in structure
        lines_with_scores = scorer(
            new_lines
        )  # This suppose to return dict of description -> (score, text)
        best_value = scorer.get_best_value()  # Text to score
        best = scorer.get_best()  # Text to (text, image)
        average_value = scorer.get_average_value()  # Text to score
        for key in average_value:
            # assert initial_scores[key] <= best_value[key][0], (initial_scores[key], best_value[key][0])
            loggers[key].write(
                f"{best_value[key][0]}\t{average_value[key]}\t{best[key]}\n"
            )
    for k, logger in loggers.items():
        logger.close()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(args.prompt, "r") as w:
        text_prompt = w.read()
    with open(args.annotations_path, "r") as w:
        annotations = json.load(w)

    clip_model, tokenizer = get_clip(args.model)
    clip_model.to(args.device)
    clip_model.eval()
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
    )
    video_ids = sorted(
        set(annotations['videos'][idx]['video_id'] for idx in range(len(annotations['videos'])))
    )
    video_ids = [
        video_id
        for video_id in video_ids
        if os.path.exists(os.path.join(args.videos_path, video_id + ".mp4"))
    ]
    # Removes videos that are already completed
    video_ids = [x for x in video_ids if not os.path.exists(os.path.join(args.output_dir, f"{x}"))]
    print(f"Length of the remaining data is {len(video_ids)}")
    if args.ablation:
        random.seed(args.seed)
        video_ids = random.sample(video_ids, 1000)

    video_ids = video_ids[args.process :: args.num_processes]
    while len(video_ids):
        current_batch = []
        while len(current_batch) < args.llm_batch_size and video_ids:
            video_id = video_ids[0]
            if (
                not os.path.exists(os.path.join(args.output_dir, f"{video_id}"))
                and video_id not in current_batch
            ):
                current_batch.append(video_id)
            video_ids = video_ids[1:]
        if current_batch:
            optimize_for_videos(
                args,
                text_pipeline,
                current_batch,
                text_prompt,
                clip_model,
                tokenizer,
            )


def get_args_parser():
    parser = argparse.ArgumentParser("Video Captioning with Vatex", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--annotations_path", default=VIDEOC_MSRVTT_ANNOTATIONS)
    parser.add_argument("--videos_path", default=VIDEOC_MSRVTT_VIDEOS)
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
    )
    parser.add_argument(
        "--model",
        default="viclip",
    )
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--process", default=0, type=int)

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--llm_batch_size", default=16, type=int, help="Batch size for llms"
    )
    parser.add_argument("--keep_previous", default=50, type=int, help="Keep previous")
    parser.add_argument(
        "--requested_number", default=50, type=int, help="How many to request"
    )
    parser.add_argument(
        "--iterations", default=10, type=int, help="Optimization iterations"
    )

    parser.add_argument(
        "--text_model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        type=str,
        help="Text model",
    )
    parser.add_argument(
        "--init_descriptions",
        default="init_descriptions/image_descriptions_per_class.txt",
        type=str,
        help="init descriptions pool",
    )
    parser.add_argument(
        "--prompt", default="prompts/video_captioning_shorter.txt", type=str, help="Prompt"
    )
    parser.add_argument("--exploration", default=0.0, type=float, help="exploration")
    # Dataset parameters
    parser.add_argument("--dataset", default='msrvtt', type=str, help="dataset name")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no_ablation", action="store_false", dest="ablation")
    parser.set_defaults(ablation=False)
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    text_model = args.text_model.split("/")[-1].replace("-", "-")
    name = "videoc_g" if not args.ablation else "videoc_a"
    args.output_dir = os.path.join(
        args.output_dir,
        f"{name}_{args.dataset}_{text_model}_{args.iterations}_{args.exploration}_{args.keep_previous}_{args.requested_number}",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
