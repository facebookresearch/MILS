# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from flask import Flask, render_template, send_from_directory
import os
import json

app = Flask(__name__)

# Path configurations
IMAGES_PATH = "/path/to/coco/val2014/val2014" # Same as IMAGEC_COCO_IMAGES in paths.py
GROUND_TRUTH_PATH = "/path/to/captions_val2014.json" # Same as IMAGEC_COCO_ANNOTATIONS

def extract_captions(folder_path, index_to_choose):
    captions = []
    for root, _, files in os.walk(folder_path):
        if 'log.txt' in files:
            with open(os.path.join(root, 'log.txt'), 'r') as f:
                caption = f.readlines()[index_to_choose].strip().split('\t')[-1]
                sample_id = os.path.basename(root)
                captions.append({"image_id": int(sample_id), "caption": caption})
    return captions

with open('/path/to/MeaCap/outputs/MeaCap__memory_cc3m_lmTrainingCorpus__0.1_0.8_0.2_k200.json') as f:
    content = json.load(f)
meacap_captions = [{"image_id": int(key[-12:]), "caption": content[key]} for key in content]

your_captions = extract_captions("/path/to/imagec_g_Meta-Llama-3.1-8B-Instruct_10_0.0_50_50_ViT-SO400M-14-SigLIP_webli_image_captioning_shorter/", -1)

# Get the 1,000 image IDs from your captions
image_ids_to_use = {item['image_id'] for item in your_captions}

# Load ground truth captions only for those image IDs
with open(GROUND_TRUTH_PATH, 'r') as f:
    coco_data = json.load(f)

gt_captions = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id in image_ids_to_use:
        if img_id not in gt_captions:
            gt_captions[img_id] = []
        gt_captions[img_id].append(ann['caption'])

# Convert lists to dictionaries for quick lookups
meacap_captions_dict = {item['image_id']: item['caption'] for item in meacap_captions}
your_captions_dict = {item['image_id']: item['caption'] for item in your_captions}

@app.route('/')
def index():
    # Collect all the necessary data to render the page
    images_data = []
    for img_id in image_ids_to_use:
        image_file = f'COCO_val2014_{str(img_id).zfill(12)}.jpg'
        gt = gt_captions.get(img_id, [])
        llava_caption = meacap_captions_dict.get(img_id, "No caption found")
        your_caption = your_captions_dict.get(img_id, "No caption found")

        images_data.append({
            'image_file': image_file,
            'gt': gt,
            'llava_caption': llava_caption,
            'your_caption': your_caption
        })

    return render_template('index2.html', images_data=images_data)

@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(IMAGES_PATH, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5012)
