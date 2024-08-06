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
from diffusers import StableDiffusion3Pipeline, FluxPipeline
import argparse
from transformers import AutoProcessor, AutoModel
import pandas as pd
import tqdm
import random
import json
from optimization_utils import Scorer, Generator, get_text_features

from paths import OUTPUT_DIR


@torch.no_grad()
def optimize_for_annotations(
    args,
    text_pipeline,
    annotations,
    text_prompt,
    post_text_function,
    image_model,
    processor,
):
    loggers = {}
    save_locations = {}
    for annotation in annotations:
        image_id = annotation[0]
        save_locations[f"{image_id}"] = os.path.join(
            args.output_dir, f"{image_id}"
        )
        os.makedirs(save_locations[f"{image_id}"], exist_ok=True)
        loggers[f"{image_id}"] = open(
            os.path.join(save_locations[f"{image_id}"], "log.txt"), "w"
        )
    generator = Generator(
        text_pipeline,
        args.text_model,
        requested_number=args.requested_number,
        keep_previous=args.keep_previous,
        prompt=text_prompt,
        key=lambda x: -x[0],
        batch_size=args.batch_size,
        device=args.device,
        post_text_function=post_text_function,
        exploration=args.exploration,
    )

    image_inputs = processor(
        text=[a[1]["Prompts"] for a in annotations],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(args.device)
    text_embs = image_model.get_text_features(**image_inputs)
    target_features = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    target_features = target_features.detach().cpu().numpy()
    assert target_features.shape[0] == len(annotations)

    def clip_scorer(images, target_feature):
        image_inputs = [processor(
            images=image[1],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )['pixel_values'].to(args.device) for image in images]
        image_inputs = {'pixel_values': torch.concatenate(image_inputs, axis=0)}
        image_embs = image_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        image_embs = image_embs.detach().cpu().numpy()
        return [float(x) for x in (image_embs @ target_feature.T)]

    scorers = {}
    for i, annotation in enumerate(annotations):
        image_id = annotation[0]
        scorers[f"{image_id}"] = {
            "func": clip_scorer,
            "target_feature": target_features[i],
        }

    scorer = Scorer(
        scorers, args.batch_size, key=lambda x: -x, keep_previous=args.keep_previous
    )
    ###
    # Initialize the pool
    ###
    init_images = post_text_function([a[1]["Prompts"] for a in annotations])
    lines_with_scores = {}
    init_descriptions = {}
    initial_scores = {}
    assert len(init_images) == len(scorer.score_functions)
    for i, annotation in enumerate(annotations):
        image_id = annotation[0]
        caption = annotation[1]["Prompts"]
        init_score = scorer.score(f"{image_id}", [(caption, init_images[i])])[0]
        initial_scores[f"{image_id}"] = init_score
        lines_with_scores[f"{image_id}"] = [
            (init_score, (caption, init_images[i]))
        ]
        loggers[f"{image_id}"].write(f"{init_score}\t{init_score}\t{caption}\n")
        init_descriptions[f"{image_id}"] = caption
        init_images[i].save(os.path.join(save_locations[f"{image_id}"], "0.png"))
    ###
    # Do the optimization:
    ###
    for it in range(args.iterations):
        new_lines = generator(lines_with_scores, init_description=init_descriptions)
        # new_lines is similar to lines in structure
        lines_with_scores = scorer(
            new_lines
        )  # This suppose to return dict of description -> (score, text)
        best_value = scorer.get_best_value()  # Text to score
        best = scorer.get_best()  # Text to (text, image)
        average_value = scorer.get_average_value()  # Text to score
        for key in lines_with_scores:
            # assert initial_scores[key] <= best_value[key][0], (initial_scores[key], best_value[key][0])
            loggers[key].write(
                f"{best_value[key][0]}\t{average_value[key]}\t{best[key][0]}\n"
            )
        for key in best:
            best[key][1].save(os.path.join(save_locations[key], f"{it+1}.png"))
        # We can save the best results
    for k, logger in loggers.items():
        logger.close()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if "FLUX" not in args.image_model:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.image_model, torch_dtype=torch.float32
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            args.image_model, torch_dtype=torch.bfloat16
        )
    pipe.enable_model_cpu_offload()
    with open(args.prompt, "r") as w:
        text_prompt = w.read()
    
    df = pd.read_csv(args.annotations_path)
    annotations = [(index, row) for (index, row) in df.iterrows()]

    @torch.no_grad()
    def post_text_function(descriptions):
        all_images = []
        # Set a fixed set of seeds.
        for d in range(0, len(descriptions), args.batch_size):
            current_desc = descriptions[d:d+args.batch_size]
            generator = [
                torch.Generator(device=args.device).manual_seed(args.seed)
                for i in range(len(current_desc))
            ]
            kwargs = {}
            if "FLUX" in args.image_model:
                kwargs["max_sequence_length"] = 256
            else:
                kwargs["negative_prompt"] = [""] * len(current_desc)

            num_inference = {
                "stabilityai/stable-diffusion-3-medium-diffusers": 28,
                "black-forest-labs/FLUX.1-schnell": 4,
            }
            images = pipe(
                prompt=current_desc,
                num_inference_steps=num_inference[args.image_model],
                height=256,
                width=256,
                guidance_scale=args.guidance_scale,
                generator=generator,
                **kwargs,
            ).images
            all_images.extend(images)
        return all_images

    processor = AutoProcessor.from_pretrained(args.processor)
    image_model = AutoModel.from_pretrained(args.image_scorer).eval().to(args.device)

    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
    )
    annotations = annotations[args.process :: args.num_processes]
    while len(annotations):
        current_batch = []
        while len(current_batch) < args.llm_batch_size and annotations:
            annotation = annotations[0]
            image_id = annotation[0]
            if not os.path.exists(os.path.join(args.output_dir, f"{image_id}")):
                current_batch.append(annotation)
            annotations = annotations[1:]
        if current_batch:
            optimize_for_annotations(
                args,
                text_pipeline,
                current_batch,
                text_prompt,
                post_text_function,
                image_model,
                processor,
            )


def get_args_parser():
    parser = argparse.ArgumentParser("Image Enhanced Generation with DrawBench", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--process", default=0, type=int)
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
    )
    parser.add_argument(
        "--annotations_path",
        default="init_descriptions/DrawBench.csv",
    )
    parser.add_argument("--exploration", default=0.0, type=float, help="exploration")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument(
        "--llm_batch_size", default=8, type=int, help="Batch size for llms"
    )
    parser.add_argument("--keep_previous", default=20, type=int, help="Keep previous")
    parser.add_argument(
        "--requested_number", default=50, type=int, help="How many to request"
    )
    parser.add_argument(
        "--iterations", default=19, type=int, help="Optimization iterations"
    )
    parser.add_argument(
        "--guidance_scale", default=7.0, type=float, help="Guidance scale"
    )
    parser.add_argument(
        "--image_scorer",
        default="yuvalkirstain/PickScore_v1",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument(
        "--processor",
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        type=str,
        help="Name of model to use",
    )
    parser.add_argument(
        "--text_model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        type=str,
        help="Text model",
    )
    parser.add_argument(
        "--prompt", default="prompts/image_generation.txt", type=str, help="Prompt"
    )
    parser.add_argument(
        "--image_model",
        default="black-forest-labs/FLUX.1-schnell",
        type=str,
        help="Image model [stabilityai/stable-diffusion-3-medium-diffusers or black-forest-labs/FLUX.1-schnell]",
    )
    # Dataset parameters
   
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    image_model = args.image_model.split("/")[-1].replace("-", "-")
    text_model = args.text_model.split("/")[-1].replace("-", "-")
    name = "image_enhance_db_g"
    args.output_dir = os.path.join(
        args.output_dir,
        f"{name}_{image_model}_{text_model}_{args.iterations}_{args.exploration}_{args.keep_previous}_{args.requested_number}_{args.batch_size}",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
