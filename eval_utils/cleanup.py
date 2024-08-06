# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Cleans a result folder by removing incomplete inference samples
'''
import argparse
import torch
import shutil
import glob
import os
import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser("Cleans a folder", add_help=False)

    # Model parameters
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--lines", required=True, type=int)
    
    return parser


def main(args):
    data = list(glob.glob(os.path.join(args.data_path, "*")))
    print(f'Found {len(data)} folders')
    for g in tqdm.tqdm(data):
        if not os.path.exists(os.path.join(g, 'log.txt')):
            print('File', os.path.join(g, 'log.txt'), 'doesn\'t exist')
            if input(f'Remove dir {g}?').strip() == 'y':
                shutil.rmtree(g)
        else:
            with open(os.path.join(g, 'log.txt'), 'r') as f:
                l = len(f.readlines())
            if l != args.lines:
                print('File', os.path.join(g, 'log.txt'), f'doesn\'t have the right number of lines ({l})')
                if input(f'Remove dir {g}?').strip() == 'y':
                    shutil.rmtree(g)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
