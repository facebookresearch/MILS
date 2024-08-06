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

def extract_captions(folder_path, index_to_choose, ablation=True):
    # If ablation is true, that means we want to see more captions per step, so show 0, 3, 7, 10
    captions = []
    for root, _, files in os.walk(folder_path):
        if 'log.txt' in files:
            with open(os.path.join(root, 'log.txt'), 'r') as f:
                file_content = f.readlines()
                if not ablation:
                    caption = file_content[index_to_choose].strip().split('\t')[-1]
                else:
                    caption = ""
                    for dummy_index in [0, 3, 7, 10]:
                        caption += f"Step {dummy_index}: {file_content[dummy_index].strip().split('\t')[-1]}\n"
                sample_id = os.path.basename(root)
                captions.append({"image_id": int(sample_id), "caption": caption})
    return captions

# Function to extract LLaVA and your captions
def extract_captions_from_results(folder):
    # Example: [{'image_id': 1, 'caption': 'A dog sitting on a couch'}, ...]
    with open(os.path.join(folder, 'captions.json')) as f:
        return json.load(f)

with open('/path/to/MeaCap/outputs/MeaCap__memory_cc3m_lmTrainingCorpus__0.1_0.8_0.2_k200.json') as f:
    content = json.load(f)
meacap_captions = [{"image_id": int(key[-12:]), "caption": content[key]} for key in content]

your_captions = extract_captions("/path/to/imagec_g_Meta-Llama-3.1-8B-Instruct_10_0.0_50_50_ViT-SO400M-14-SigLIP_webli_image_captioning_shorter/", -1)

# Get the 1,000 image IDs from your captions
image_ids_to_use = {item['image_id'] for item in your_captions}

# Load ground truth captions only for those image_ids
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
llava_captions_dict = {item['image_id']: item['caption'] for item in meacap_captions}
your_captions_dict = {item['image_id']: item['caption'] for item in your_captions}

@app.route('/')
def index():
    # Only display the 1,000 relevant image IDs
    image_ids = list(image_ids_to_use)
    return render_template('index.html', image_ids=image_ids)

@app.route('/image/<int:image_id>')
def show_image(image_id):
    # Find the required data for the image
    image_file = f'COCO_val2014_{str(image_id).zfill(12)}.jpg'
    image_path = os.path.join(IMAGES_PATH, image_file)
    
    # Fetch captions
    gt = gt_captions.get(image_id, [])
    llava_caption = llava_captions_dict.get(image_id, "No caption found")
    your_caption = your_captions_dict.get(image_id, "No caption found")

    return render_template('image_view.html', 
                           image_file=image_file, 
                           gt=gt, 
                           llava_caption=llava_caption, 
                           your_caption=your_caption)

@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(IMAGES_PATH, filename)

if __name__ == '__main__':
    app.run(port=5006, debug=True)
