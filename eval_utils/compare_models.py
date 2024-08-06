# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import matplotlib.pyplot as plt
import argparse
import glob
import numpy as np
import random
import json

def get_args_parser():
    parser = argparse.ArgumentParser("Simple summarization and graphs script", add_help=False)

    # Model parameters
    parser.add_argument("--data_path1", default='/path/to/output/dir/1/')
    parser.add_argument("--data_path2", default='/path/to/output/dir/2/')
    parser.add_argument(
        "--output_dir",
        default='../output_dir',
        help="Output Path",
    )
    parser.add_argument('--seed', default=2024)
    return parser


def get_caption(image_id, annotations):
    x = []
    for a in annotations:
        if a['image_id'] == image_id:
            x.append(a['caption'])
    return x

def main(args):
    for folder in glob.glob(os.path.join(args.data_path2, '*')):
        folder = os.path.split(folder)[-1]
        with open(os.path.join(args.data_path1, folder, 'log.txt'), 'r') as r:
            lines = [x.strip() for x in r.readlines()]
            print('A:', lines[-1].split('\t')[-1])
        with open(os.path.join(args.data_path2, folder, 'log.txt'), 'r') as r:
            lines = [x.strip() for x in r.readlines()]
            print('B:', lines[-1].split('\t')[-1])
        print()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
