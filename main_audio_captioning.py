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
import torchvision.transforms as transforms
import argparse
import tqdm
import random
import json
import pandas as pd
from random import sample
from optimization_utils import (
    Scorer as S,
    Generator as G,
    get_text_features,
    get_image_features,
)
from task_utils.audio import imagebind_model
from task_utils.audio.data import load_and_transform_audio_data, load_and_transform_text
from task_utils.audio.imagebind_model import ModalityType

from paths import AUDIOC_CLOTHO_ANNOTATIONS, AUDIOC_CLOTHO_FILES, OUTPUT_DIR


# args, text_pipeline, current_batch, text_prompt, model
def optimize_for_images(args, text_pipeline, questions, text_prompt, image_bind):
    loggers = {}
    save_locations = {}
    for idx, file_name in questions:
        save_locations[f"{idx}"] = os.path.join(args.output_dir, f"{idx}")
        os.makedirs(save_locations[f"{idx}"], exist_ok=True)
        loggers[f"{idx}"] = open(os.path.join(save_locations[f"{idx}"], "log.txt"), "w")
    generator = G(
        text_pipeline,
        args.text_model,
        requested_number=args.requested_number,
        keep_previous=args.keep_previous,
        prompt=text_prompt,
        key=lambda x: -x[0],
        batch_size=args.batch_size,
        device=args.device,
    )
    audio_paths = [
        os.path.join(args.audio_files, file_name) for (q, file_name) in questions
    ]
    inputs = {
        ModalityType.AUDIO: load_and_transform_audio_data(audio_paths, args.device),
    }
    with torch.no_grad():
        target_features = image_bind(inputs)[ModalityType.AUDIO].detach().cpu().numpy()
    torch.cuda.empty_cache()

    def clip_scorer(sentences, target_feature):
        for s in sentences:
            assert "<|endoftext|>" not in s, s
        text_features_list = []
        for batch in range(0, len(sentences), args.batch_size):
            inputs = {
                ModalityType.TEXT: load_and_transform_text(
                    sentences[batch : batch + args.batch_size], args.device
                ),
            }
            with torch.no_grad():
                text_features = (
                    image_bind(inputs)[ModalityType.TEXT].detach().cpu().numpy()
                )
            text_features_list.append(text_features)
        return np.concatenate(text_features_list, axis=0) @ target_feature

    scorers = {}
    for i, (idx, file_name) in enumerate(questions):
        scorers[f"{idx}"] = {"func": clip_scorer, "target_feature": target_features[i]}

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
    all_idx = []
    for i, (idx, file_name) in enumerate(questions):
        init_scores = scorer.score(f"{idx}", init_sentences)
        all_idx.append(f"{idx}")
        lines_with_scores[f"{idx}"] = [
            (s, l) for (s, l) in zip(init_scores, init_sentences)
        ]
        best_score = sorted(lines_with_scores[f"{idx}"], key=lambda x: -x[0])[0]
        initial_scores[f"{idx}"] = best_score
        mean_score = np.mean(init_scores)
        bs = best_score[1].strip()
        loggers[f"{idx}"].write(f"{best_score[0]}\t{mean_score}\t{bs}\n")

    ###
    # Do the optimization:
    ###
    for it in range(args.iterations):
        torch.cuda.empty_cache()
        new_lines = generator(lines_with_scores, examples=_format_examples(all_idx, init_sentences))
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

def _format_examples(all_idx, init_sentences, l=5):
    examples = {}
    for idx in all_idx:
        data = sample(init_sentences, l)
        examples[idx] = ""
        for i, d in enumerate(data):
            examples[idx] += f'{i+1}: {d}\n'
    return examples


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(args.prompt, "r") as w:
        text_prompt = w.read()
    df = pd.read_csv(args.annotations)

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(args.device)

    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
    )

    annotations = [(index, row) for (index, row) in df.iterrows()]

    annotations = annotations[args.process :: args.num_processes]
    while len(annotations):
        current_batch = []
        while len(current_batch) < args.llm_batch_size and annotations:
            idx, row = annotations[0]
            if (
                not os.path.exists(os.path.join(args.output_dir, f"{idx}"))
                and (idx, row["file_name"]) not in current_batch
            ):
                current_batch.append((idx, row["file_name"]))
            annotations = annotations[1:]
        if current_batch:
            optimize_for_images(args, text_pipeline, current_batch, text_prompt, model)


def get_args_parser():
    parser = argparse.ArgumentParser("Audio Captioning", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument(
        "--annotations",
        default=AUDIOC_CLOTHO_ANNOTATIONS,
        help="Annotations",
    )
    parser.add_argument(
        "--audio_files",
        default=AUDIOC_CLOTHO_FILES,
        help="Annotations",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
    )
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--process", default=0, type=int)
    
    parser.add_argument(
        "--prompt",
        default="prompts/audio_captioning_shorter.txt",
        help="The captioning instruction",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--llm_batch_size", default=8, type=int, help="Batch size for llms"
    )

    parser.add_argument("--keep_previous", default=10, type=int, help="Keep previous")
    parser.add_argument(
        "--requested_number", default=50, type=int, help="How many to request"
    )
    parser.add_argument(
        "--iterations", default=19, type=int, help="Optimization iterations"
    )
    # Dataset parameters
    parser.add_argument(
        "--text_model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        type=str,
        help="Text model",
    )
    parser.add_argument(
        "--init_descriptions",
        default="init_descriptions/audio_descriptions_per_class_405_v2.txt",
        type=str,
        help="init descriptions pool",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    text_model = args.text_model.split("/")[-1].replace("-", "-")
    prompt = args.prompt.split("/")[-1].split(".")[0]
    init_descriptions = args.init_descriptions.split("/")[-1].split(".")[0]
    args.output_dir = os.path.join(
        args.output_dir,
        f"audio_{text_model}_{args.iterations}_{args.keep_previous}_{args.requested_number}_{init_descriptions}_{prompt}",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
