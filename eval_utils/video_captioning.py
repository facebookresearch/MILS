# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from collections import defaultdict
from pycocoevalcap.eval import COCOEvalCap
import tempfile
from audio_captioning import EvalCap



def extract_captions(folder_path, index_to_choose):
    captions = {}
    for root, _, files in os.walk(folder_path):
        if 'log.txt' in files:
            with open(os.path.join(root, 'log.txt'), 'r') as f:
                file_content = f.readlines()
                if len(file_content) != 11:
                    continue
                caption = file_content[index_to_choose].strip().split('\t')[-1]
                sample_id = os.path.basename(root)
                captions[sample_id] = [caption]
    return captions


from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

method = "ours"
if method == "ours":
    ours_result_path = '/path/to/output/dir/'
    result_data = extract_captions(ours_result_path, -1)

ref_dict = {}
annotation_file = '/path/to/msrvtt/test_videodatainfo.json' # Same as paths.VIDEOC_MSRVTT_ANNOTATIONS
with open(annotation_file) as f:
    annot = json.load(f)
for idx in range(len(annot['sentences'])):
    if annot['sentences'][idx]['video_id'] not in ref_dict:
        ref_dict[annot['sentences'][idx]['video_id']] = []
    ref_dict[annot['sentences'][idx]['video_id']].append(annot['sentences'][idx]['caption'])

eval_scorer = EvalCap(result_data, ref_dict)

metrics = eval_scorer.compute_scores()
print(metrics)
