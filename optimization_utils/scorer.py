# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os.path
import datetime
import json
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
from typing import Any, Text, Dict, List


def get_text_features(
    model, tokenizer, lines, device, batch_size, amp=True, use_format=False
):
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
    with torch.no_grad():
        zeroshot_weights = []
        for i in range(0, len(lines), batch_size):
            texts = lines[i : i + batch_size]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            zeroshot_weights.append(class_embeddings.detach())
        zeroshot_weights = torch.concatenate(zeroshot_weights, dim=0)
    return zeroshot_weights


@torch.no_grad()
def get_image_features(
    model, preprocess, image_paths, device, batch_size
):
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
    zeroshot_weights = []
    for i in tqdm.trange(0, len(image_paths), batch_size):
        torch_images = []
        for j in image_paths[i : i + batch_size]:
            im = Image.open(j)
            torch_images.append(preprocess(im).to(device, torch.float))
            im.close()
        torch_images = torch.stack(torch_images, axis=0)
        zeroshot_weights.append(model.encode_image(torch_images, normalize=True))
    return torch.concatenate(zeroshot_weights, axis=0)



class Scorer(object):
    def __init__(self, score_functions, batch_size, key, keep_previous):
        self.score_functions = score_functions
        self.batch_size = batch_size
        self.best = {}
        self.keep_previous = keep_previous
        self.best_value = {}
        self.average_value = {}
        self.key = key

    def __call__(self, inputs: Dict[Text, List[Any]]):
        # run the score function on the inputs, in a batched way
        new_d = {}
        self.best = {}
        self.best_value = {}
        self.average_value = {} 
        for d, pils in inputs.items():
            new_d[d] = self._run_score_function(d, pils)
            argmin = np.argmin([self.key(x[0]) for x in new_d[d]])
            self.best[d] = pils[argmin]
            self.best_value[d] = new_d[d][argmin]
            average_data = np.array([x[0] for x in new_d[d]])
            self.average_value[d] = np.mean(np.sort([self.key(i) for i in average_data])[:self.keep_previous])
        return new_d

    def _run_score_function(self, d_id, data):
        results = []
        for i in range(0, len(data), self.batch_size):
            # TODO: you can cache here results from previous runs
            current_batch = data[i:i+self.batch_size]
            scores = self.score(d_id, current_batch)
            # scores might be one value if the len of the input is 1, or more.
            if hasattr(scores[0], 'shape') and len(scores[0].shape) != 0:
                for j in range(len(current_batch)):
                    current_score = tuple([s[j] for s in scores])
                    results.append((current_score, current_batch[j]))
            else:
                assert len(scores) == len(current_batch), (len(scores), len(current_batch))
                results += [(s, pil) for (s, pil) in zip(scores, current_batch)]
        assert len(results) == len(data)
        return results

    def score(self, d_id, data):
        func = self.score_functions[d_id]['func']
        kwargs = {i: v for (i, v) in self.score_functions[d_id].items() if i != 'func'}
        return func(data, **kwargs)

    def get_best(self):
        return self.best

    def get_best_value(self):
        return self.best_value

    def get_average_value(self):
        return self.average_value
