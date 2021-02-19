# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)
from torch.utils.data.dataloader import DataLoader

import dataset
import model
import trainer
import utils
import math
import logging

logger = logging.getLogger(__name__)

argp = argparse.ArgumentParser()
argp.add_argument('function',
    help="Whether to pretrain, finetune or evaluate a model",
    choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
    help="Which variant of the model to run ('vanilla' or 'synthesizer')",
    choices=["vanilla", "synthesizer"])
argp.add_argument('pretrain_corpus_path',
    help="Path of the corpus to pretrain on", default=None)
argp.add_argument('--reading_params_path',
    help="If specified, path of the model to load before finetuning/evaluation",
    default=None)
argp.add_argument('--writing_params_path',
    help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
    help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()

if 1==True:
    assert args.outputs_path is not None
    # assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    # model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w',encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path,encoding='utf-8')):
            # x = line.split('\t')[0]
            # x = x + '⁇'
            # x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
            # pred = utils.sample(model, x, 32, sample=False)[0]
            # completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = "London"
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))
