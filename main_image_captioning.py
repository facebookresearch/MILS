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
import matplotlib.pyplot as plt
import numpy as np
import transformers
import torchvision.transforms as transforms
import argparse
import tqdm
import random
import json
from open_clip import create_model_and_transforms, get_tokenizer
from optimization_utils import (
    Scorer as S,
    Generator as G,
    get_text_features,
    get_image_features,
)

from paths import IMAGEC_COCO_ANNOTATIONS, IMAGEC_COCO_IMAGES, IMAGEC_COCO_SPLITS, OUTPUT_DIR


def optimize_for_images(
    args, text_pipeline, image_ids, text_prompt, clip_model, tokenizer, preprocess
):
    loggers = {}
    save_locations = {}
    for image_id in image_ids:
        save_locations[f"{image_id}"] = os.path.join(args.output_dir, f"{image_id}")
        os.makedirs(save_locations[f"{image_id}"], exist_ok=True)
        loggers[f"{image_id}"] = open(
            os.path.join(save_locations[f"{image_id}"], "log.txt"), "w"
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
    image_paths = [
        os.path.join(args.images_path, f"COCO_val2014_{image_id:012}.jpg")
        for image_id in image_ids
    ]
    target_features = (
        get_image_features(
            clip_model,
            preprocess,
            image_paths,
            args.device,
            args.batch_size,
        )
        .detach()
        .cpu()
        .numpy()
    )

    def clip_scorer(sentences, target_feature):
        text_features = get_text_features(
            clip_model,
            tokenizer,
            sentences,
            args.device,
            args.batch_size,
            amp=True,
            use_format=False,
        )
        return text_features.detach().cpu().numpy() @ target_feature

    scorers = {}
    for i, image_id in enumerate(image_ids):
        scorers[f"{image_id}"] = {
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
    if args.init_descriptions_set_size != "all":
        random.seed(0) # Choose a different seed than args as it is already used (should not matter though)
        init_sentences = random.sample(init_sentences, int(args.init_descriptions_set_size)) # Must be all or an int
    lines_with_scores = {}
    initial_scores = {}
    for i, image_id in enumerate(image_ids):
        init_scores = scorer.score(f"{image_id}", init_sentences)
        lines_with_scores[f"{image_id}"] = [
            (s, l) for (s, l) in zip(init_scores, init_sentences)
        ]
        best_score = sorted(lines_with_scores[f"{image_id}"], key=lambda x: -x[0])[0]
        initial_scores[f"{image_id}"] = best_score
        mean_score = np.mean(init_scores)
        bs = best_score[1].strip()
        loggers[f"{image_id}"].write(f"{best_score[0]}\t{mean_score}\t{bs}\n")
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
        annotations = json.load(w)["annotations"]

    clip_model, _, preprocess = create_model_and_transforms(
        args.clip_model, pretrained=args.pretrained
    )
    tokenizer = get_tokenizer(args.clip_model)
    clip_model.to(args.device)
    clip_model.eval()
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
    )
    if 'Ministral' in args.text_model:
        text_pipeline.tokenizer.pad_token_id = text_pipeline.model.config.eos_token_id
    image_ids = sorted(set(int(a["image_id"]) for a in annotations))
    # Choose karpathy test set splits
    with open(IMAGEC_COCO_SPLITS) as f:
        karpathy_split = json.load(f)['images']
    image_ids = [int(x['filename'].split('CO_val2014_')[-1].split('.jpg')[0]) for x in karpathy_split if x['split'] == 'test']
    # Sample 1000 if ablation
    if args.ablation:
        random.seed(args.seed)
        image_ids = random.sample(image_ids, 1000)
    image_ids = [x for x in image_ids if not os.path.exists(os.path.join(args.output_dir, f"{x}"))]
    print(f"Length of the data is {len(image_ids)}")


    image_ids = image_ids[args.process :: args.num_processes]
    while len(image_ids):
        current_batch = []
        while len(current_batch) < args.llm_batch_size and image_ids:
            image_id = image_ids[0]
            if (
                not os.path.exists(os.path.join(args.output_dir, f"{image_id}"))
                and image_id not in current_batch
            ):
                current_batch.append(image_id)
            image_ids = image_ids[1:]
        if current_batch:
            optimize_for_images(
                args,
                text_pipeline,
                current_batch,
                text_prompt,
                clip_model,
                tokenizer,
                preprocess,
            )


def get_args_parser():
    parser = argparse.ArgumentParser("Image Captioning with COCO", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument(
        "--annotations_path",
        default=IMAGEC_COCO_ANNOTATIONS,
    )
    parser.add_argument(
        "--images_path", default=IMAGEC_COCO_IMAGES
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
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
        "--clip_model",
        default="ViT-SO400M-14-SigLIP",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="webli", type=str)

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
        "--init_descriptions_set_size",
        default="all",
        type=str,
        help="How many descriptions to choose, should be either int or an int",
    )
    parser.add_argument(
        "--prompt", default="prompts/image_captioning_shorter.txt", type=str, help="Prompt"
    )
    parser.add_argument("--exploration", default=0.0, type=float, help="exploration")
    # Dataset parameters
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no_ablation", action="store_false", dest="ablation")
    parser.set_defaults(ablation=False)
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    text_model = args.text_model.split("/")[-1].replace("-", "-")
    prompt = args.prompt.split("/")[-1].split(".")[0]
    name = "imagec_g" if not args.ablation else "imagec_a"
    args.output_dir = os.path.join(
        args.output_dir,
        f"{name}_{text_model}_{args.iterations}_{args.exploration}_{args.keep_previous}_{args.requested_number}_{args.clip_model}_{args.pretrained}_{prompt}_{args.init_descriptions_set_size}",
    )
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
