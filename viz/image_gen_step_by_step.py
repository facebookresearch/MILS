# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)
main_folder = '/path/to/image_enhance_db_g_stable-diffusion-3-medium-diffusers_Meta-Llama-3.1-8B-Instruct_19_0.0_20_50_64/'  # Replace with the path to your main folder containing folders 0-199

@app.route('/')
def home():
    folders = []
    
    # Gather images and selected log lines from each folder
    for folder_id in range(200):
        folder_data = {'folder_id': folder_id, 'images': [], 'log_lines': {}}
        folder_path = os.path.join(main_folder, str(folder_id))
        
        # Add specified images if they exist
        for index in [0, 3, 7, 11, 15, 19]:
            image_name = f"{index}.png"
            image_path = os.path.join(folder_path, image_name)
            if os.path.exists(image_path):
                folder_data['images'].append({
                    'index': index,
                    'url': f"/static/{folder_id}/{image_name}"
                })
        
        # Add selected lines (0th, 3rd, 11th, 15th, 19th) from log.txt if it exists
        log_path = os.path.join(folder_path, 'log.txt')
        if os.path.exists(log_path):
            with open(log_path, 'r') as log_file:
                log_lines = log_file.readlines()
                folder_data['log_lines'] = {
                    0: log_lines[0].strip().split('\t')[-1] if len(log_lines) > 0 else "",
                    3: log_lines[3].strip().split('\t')[-1] if len(log_lines) > 3 else "",
                    7: log_lines[7].strip().split('\t')[-1] if len(log_lines) > 7 else "",
                    11: log_lines[11].strip().split('\t')[-1] if len(log_lines) > 11 else "",
                    15: log_lines[15].strip().split('\t')[-1] if len(log_lines) > 15 else "",
                    19: log_lines[19].strip().split('\t')[-1] if len(log_lines) > 19 else ""
                }

        folders.append(folder_data)

    return render_template('image_gen_progress.html', folders=folders)

# Route to serve images from the folders
@app.route('/static/<folder_id>/<filename>')
def static_files(folder_id, filename):
    return send_from_directory(os.path.join(main_folder, folder_id), filename)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
