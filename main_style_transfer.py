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
from diffusers import StableDiffusion3Pipeline
import argparse
import tqdm
from optimization_utils import Scorer as S, Generator as G
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from task_utils.style_transfer import get_style_and_content_losses, image_loader, loader

from paths import OUTPUT_DIR

def _log_for_debug(log, images, style_score, content_score):
    for i, s, c in zip(images, style_score, content_score):
        log.write(f'{i[0]}, {s}, {c}\n')


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = open(os.path.join(args.output_dir, 'log.txt'), 'w')
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    with open(args.init_descriptions, 'r') as w:
        init_descriptions = [i.strip() for i in w.readlines()]

    with open(args.prompt, 'r') as w:
        text_prompt = w.read()

    content_img = image_loader(args.content_image).to(args.device)
    style_img = image_loader(args.style_image).to(args.device)

    feature_extractor ,style_losses, content_losses = get_style_and_content_losses(args, style_img, content_img)
    
    def post_text_function(descriptions):
        generator = [torch.Generator(device=args.device).manual_seed(args.seed) for i in range(len(descriptions))]
        images = pipe(descriptions, image=content_img, generator=generator).images
        return images
    
    # log = open('a.txt', 'w')
    
    @torch.no_grad()
    def style_transfer_scorer(images):
        assert len(images) <= args.style_batch_size, len(images)
        torch_images = []
        for text, image in images:
            image = loader(image)
            torch_images.append(image.to(torch.float))
        torch_images = torch.stack(torch_images, axis=0).to(args.device)
        feature_extractor(torch_images.to(args.device, torch.float))
        style_score = 0
        content_score = 0
        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
        # _log_for_debug(log, images, style_score, content_score)
        assert style_score.shape[0] == torch_images.shape[0], (style_score.shape, torch_images.shape)
        assert content_score.shape[0] == torch_images.shape[0], (content_score.shape, torch_images.shape)
        return style_score.detach().cpu().numpy() * args.style_coeff, content_score.detach().cpu().numpy()
    
    
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
        batch_size=args.batch_size,
    )

    generator = G(text_pipeline, 
                  args.text_model, 
                  requested_number=args.requested_number, 
                  keep_previous=args.keep_previous, 
                  prompt=text_prompt, 
                  key=lambda x: x[0][0] + x[0][1],
                  batch_size=args.batch_size,
                  device=args.device, 
                  post_text_function=post_text_function)
    
    scorer = S({'style transfer': {'func': style_transfer_scorer}}, 
               args.style_batch_size, key=lambda x: x[0] + x[1], keep_previous=args.keep_previous)
    ### 
    # Initialize the pool
    ###

    init_images = []
    for i in range(0, len(init_descriptions), args.batch_size):
        init_images += post_text_function(init_descriptions[i: i+args.batch_size])
    init_descriptions_with_images = [(init_description, image) for (init_description, image) in zip(init_descriptions, init_images)]
    lines_with_scores = scorer({'style transfer': init_descriptions_with_images})
    best_value = scorer.get_best_value()['style transfer'] # Text to score
    best = scorer.get_best()['style transfer'] # Text to (text, image)
    average_value = scorer.get_average_value()['style transfer'] # Text to score
    logger.write(f'{best_value[0]}\t{average_value}\t{best[0]}\n')
    best[1].save(os.path.join(args.output_dir, f'0.png'))
    ### 
    # Do the optimization:
    ###
    for it in range(args.iterations):
        # log.write(f'------\n')
        new_lines = generator(lines_with_scores)
        # new_lines is similar to lines in structure
        lines_with_scores = scorer(new_lines) # This suppose to return dict of description -> (score, text)
        best_value = scorer.get_best_value()['style transfer']  # Text to score
        best = scorer.get_best()['style transfer']   # Text to (text, image)
        average_value = scorer.get_average_value()['style transfer'] # Text to score
        logger.write(f'{best_value[0]}\t{average_value}\t{best[0]}\n')
        best[1].save(os.path.join(args.output_dir, f'{it+1}.png'))
        # We can save the best results
    logger.close()

def get_args_parser():
    parser = argparse.ArgumentParser("Image Generation", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--style_image", default="images/Henri-Matisse.jpg", help="")
    parser.add_argument("--content_image", default="images/monalisa.jpg", help="")

    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
    )
    parser.add_argument(
        "--init_descriptions",
        default='init_descriptions/pix2pix_descriptions.txt',
        type=str,
        help="init descriptions pool"
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help='Batch size'
    )
    parser.add_argument(
        "--style_batch_size", default=2, type=int, help='Batch size'
    )
    parser.add_argument(
        "--keep_previous", default=100, type=int, help='Keep previous'
    )
    parser.add_argument(
        "--requested_number", default=1000, type=int, help='How many to request'
    )
    parser.add_argument(
        "--style_coeff", default=1000., type=int, help='What is the style importance'
    )
    parser.add_argument(
        "--iterations", default=20, type=int, help='Optimization iterations'
    )
    parser.add_argument(
        "--text_model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        type=str,
        help='Text model'
    )
    parser.add_argument(
        "--prompt",
        default='prompts/style_transfer.txt',
        type=str,
        help='Prompt'
    )
    # Dataset parameters
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    style_image_text = os.path.split(args.style_image)[-1].split('.')[0]
    content_image_text = os.path.split(args.content_image)[-1].split('.')[0]
    args.output_dir = os.path.join(args.output_dir, f'style_transfer_{content_image_text}_{style_image_text}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
