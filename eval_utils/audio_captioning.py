# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2023-2024 Xiaomi Corporation and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import pandas as pd
import numpy as np
import os.path
# from utils.fense.evaluator import Evaluator


class EvalCap:
    def __init__(self, predicted, original):
        self.audioIds = sorted(predicted.keys())  # sort key for metrics require
        self.predicted = {key: predicted[key] for key in self.audioIds}
        self.original = {key: original[key] for key in self.audioIds}
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]
        # self.fense_scorer = Evaluator(
        #     device="cpu",
        #     sbert_model="paraphrase-TinyBERT-L6-v2",
        #     echecker_model="echecker_clotho_audiocaps_base",
        # )

    def compute_scores(self):
        total_scores = {}
        for score_class, method in self.scorers:
            print(score_class)
            score, scores = score_class.compute_score(self.original, self.predicted)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    total_scores[m] = {"score": sc, "scores": scs}
            else:
                if method == "SPICE":
                    spice_scores = []
                    for ss in scores:
                        spice_scores.append(ss["All"]["f"])
                    total_scores[method] = {"score": score, "scores": spice_scores}
                else:
                    total_scores[method] = {"score": score, "scores": scores}
        return_dict = {
            "bleu_1": total_scores["Bleu_1"]["score"],
            "bleu_2": total_scores["Bleu_2"]["score"],
            "bleu_3": total_scores["Bleu_3"]["score"],
            "bleu_4": total_scores["Bleu_4"]["score"],
            "meteor": total_scores["METEOR"]["score"],
            "rouge_l": total_scores["ROUGE_L"]["score"],
            "cider": total_scores["CIDEr"]["score"],
            "spice": total_scores["SPICE"]["score"],
            "spider": (
                (total_scores["CIDEr"]["score"] + total_scores["SPICE"]["score"]) / 2
            ),
            "data": [],
        }

        fense_score_list = []
        bert_score_list = []
        spider_fl_score_list = []
        return return_dict


if __name__ == "__main__":
    df = pd.read_csv('/path/to/Clotho2/clotho_captions_evaluation.csv') # same as paths.AUDIOC_CLOTHO_ANNOTATIONS
    annotations = [(index, row) for (index, row) in df.iterrows()]
    ref_dict = {
        an['file_name']: [an['caption_1'], an['caption_2'],an['caption_3'],an['caption_4'],an['caption_5']] for (i, an) in annotations
    }
    assert all([min([len(x) for x in curr_cap]) > 10 for curr_cap in list(ref_dict.values())]), "Seems like some caption are empty!"
    predict_dict = {}
    diff_format = {f'{index}': row['file_name'] for (index, row) in annotations}
    address = '/path/to/output/dir/'
    all_clip = []
    # Take sys.argv for ablation
    try:
        import sys
        index_to_choose = int(sys.argv[1])
    except:
        index_to_choose = -1 # last line by default
    for i, k in diff_format.items():
        try: 
            with open(os.path.join(address, i, 'log.txt'), 'r') as w:
                lines = w.readlines()
                predict_dict[k] = [lines[index_to_choose].split('\t')[2].strip()]
                all_clip.append(float(lines[-1].split('\t')[0]))
            print('-----')
            print(k)
            print(predict_dict[k])
            print(ref_dict[k])
        except Exception as e:
            print(e, i)
    print(len(predict_dict))
    print('CLIP:', np.mean(all_clip))
    eval_scorer = EvalCap(predict_dict, ref_dict)

    metrics = eval_scorer.compute_scores()
    print(metrics)
