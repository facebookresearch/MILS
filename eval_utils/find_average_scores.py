# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
This code reads all the log.txt and get a plot for score w.r.t. steps
'''
import os

# DrawBench
# num_steps = 20
# dataset_samples = 200
# offset = 0
# MS COCO Karpathy split
num_steps = 11
dataset_samples = 1000# 5000
offset = 0 
output_root = '/path/to/output/folder/'

output_dirs = [x for x in os.listdir(output_root) if 'log.txt' in os.listdir(os.path.join(output_root, x))]
ablation = True
if ablation and len(output_dirs) > 1000:
    # Rather than sampling 1000 again, just use the IDs from a sample output list
    output_dirs = [x for x in output_dirs if x in os.listdir('/path/to/a/sample/output/')]

# assert len(output_dirs) == dataset_samples, "Invalid count for drawbench"
dataset_samples = len(output_dirs)

per_step_average = [0.] * num_steps
for output_idx in output_dirs:
    with open(os.path.join(output_root, output_idx, 'log.txt')) as f:
        logs = f.readlines()
        # assert len(logs) == num_steps+offset, "Incomplete file"
    for line_idx in range(offset, num_steps + offset):
        avg_val = float(logs[line_idx].split('\t')[1])
        per_step_average[line_idx-offset] += avg_val

per_step_average = [-1.*x / dataset_samples for x in per_step_average]
print(per_step_average)
for step_idx, avg_val in enumerate(per_step_average):
    print(f"({step_idx}, {per_step_average[step_idx]})")