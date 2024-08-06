# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
This code deletes all folders in the output dir that does not have the required number of lines
'''
import os
import shutil


result_path = "/path/to/output/dir/"
images = os.listdir(result_path)
removed_count = 0
NUM_ROWS = 11 # Important to set this correctly

for image in images:
    if os.path.exists(os.path.join(result_path, image, 'log.txt')):
        with open(os.path.join(result_path, image, 'log.txt')) as f:
            captions = f.readlines()
        if len(captions) != NUM_ROWS:
            print(f"Error: {image}")
            removed_count += 1
            shutil.rmtree(os.path.join(result_path, image))
    else:
        print(f"Error: {image}")
        removed_count += 1
        shutil.rmtree(os.path.join(result_path, image))

print(f"{removed_count} files removed...")