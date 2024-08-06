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



def extract_captions(folder_path, index_to_choose):
    captions = []
    for root, _, files in os.walk(folder_path):
        if 'log.txt' in files:
            with open(os.path.join(root, 'log.txt'), 'r') as f:
                caption = f.readlines()[index_to_choose].strip().split('\t')[-1]
                # if len(caption) > 500:
                #     caption = caption[:500] # SPICE fails when LLM output is incorrect
                sample_id = os.path.basename(root)
                captions.append({"image_id": int(sample_id), "caption": caption})
    return captions


from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

annotation_file = '/path/to/coco/val2014/annotations/captions_val2014.json' # Same as IMAGEC_COCO_ANNOTATIONS
method = "ours"
if method == "ours":
    ours_result_path = '/path/to/output/dir/'

    # Take sys.argv for ablation
    try:
        import sys
        index_to_choose = int(sys.argv[1])
    except:
        index_to_choose = -1 # last line by default
    result_data = extract_captions(ours_result_path, index_to_choose)
elif method == "meacap":
    meacap_result_path = '/path/to/output/dir/MeaCap__memory_cc3m_lmTrainingCorpus__0.1_0.8_0.2_k200.json'
    with open(meacap_result_path) as f:
        meacap_result = json.load(f)
    result_data = []
    for key in meacap_result:
        result_data.append({'image_id': int(key[-12:]), 'caption': meacap_result[key]})

ablation = False
if ablation and len(result_data) > 1000:
    # For ablation we choose 1k video subset. The subset is determined by random seed. Now take any ablation folder to get the video list
    result_data = [x for x in result_data if x['image_id'] in [int(y) for y in os.listdir('/path/to/ablation/output/')]]
    assert len(result_data) == 1000, "Test set is not consistent"

print("$"*100)
for key in range(len(result_data)):
    print(result_data[key])
print("$"*100)

# create coco object and coco_result object
coco = COCO(annotation_file)
# coco_result = coco.loadRes(results_file)

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as temp_file:
    json.dump(result_data, temp_file)  # Write the dictionary to the temp file
    temp_file.flush()  # Ensure all data is written to the file
    temp_file_path = temp_file.name  # Get the file path

    coco_result = coco.loadRes(temp_file_path)



# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.6f}')
