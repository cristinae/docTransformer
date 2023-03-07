# -*- coding: utf-8 -*-

import warnings
import argparse
import os.path
import logging

from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
import trainer

import numpy as np
import random


logger = logging.getLogger(__name__)

    
def readCommandLine():
    ''' Parser for command line arguments'''
    
    def boolean_string(s):
        if s not in {'False', 'True'}:
           raise ValueError('Not a valid boolean string')
        return s == 'True'
        
    parser = argparse.ArgumentParser(description="Fine-tuning Roberta for document classification")

    # Input/Output
    parser.add_argument("-c", "--train_dataset", required=False, type=str, default='./train10000', help="Training dataset for classification")
    parser.add_argument("-v", "--validation_dataset", type=str, default='./dev10000', help="Validation set")
    parser.add_argument("-t", "--test_dataset", type=str, default='./corpus/corpus.right.elimparcial.txt', help="Test set to evaluate or classify")
    parser.add_argument("-o", "--classification_model", required=False, type=str, default='./model/model.monob6.bin', help="Name for the model file")
    parser.add_argument("--buffer", type=int, default=1000, help="Test documents are not loaded completely into memory but into <buffer> chunks. Default: 100000 documents.")
    parser.add_argument("--shuffling", type=boolean_string, default=True, help="Suffling within a dataset buffer. Options: True, False. Default: True.")
    
    # Task (training by default)
    parser.add_argument("--task", type=str, default='training', help="Task to perform. Options: training, evaluation, classification. Default: training.")
    
    # Data processing
    # Tokenisation
    parser.add_argument("--truncation", type=boolean_string, default=True, help="")
    parser.add_argument("--padding", type=boolean_string, default=True, help="")
    parser.add_argument("--max_length", type=int, default=None, help="")
    
    parser.add_argument("--split_documents", type=boolean_string, default=False, help="Deal with documents as a set of sentences")
    parser.add_argument("--sentence_delimiter", type=str, default='<NS>', help="Delimiter used to sepatate fragments in a document")
    parser.add_argument("--split_method", type=str, default='sentence', help="How to split the document. Options: char (exact splitting at char level), sentence (aprox splitting at sentence level). Default: sentence.")    

    # Base model
    parser.add_argument("-m", "--pretrained_model", type=str, default='skimai/spanberta-base-cased', help="pretrained model (currently only Roberta family implemented)")
    parser.add_argument("-f", "--freeze_pretrained", type=boolean_string, default=False, help="Freeze weights of the pretrained model. Options: True, False")
    
    # Training
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=6, help="Number of documents in a batch. Default: 1")
    parser.add_argument("--gradient_accumulation_size", type=int, default=1, help="Creating a larger effective batch size by accumulating n batches before updating. Default: 1")
    parser.add_argument("--sentence_batch_size", type=int, default=12, help="Number of sentences per document in a batch. Default: 24.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of batches prior to validation. Default: 100")

    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW. Default: 2e-5.")
    parser.add_argument("--dropout_prepooling", type=float, default=0.1, help="Dropout to be applied after retrieving the embeddings and before pooling. Default: 0.1.")

    # Classifier
    parser.add_argument("-d", "--dropout_postpooling", type=float, default=0.1, help="Dropout to be applied before the last linear layer of the classifier. Default: 0.1.")
    parser.add_argument("--input_to_classifier", type=str, default='cls_tanh', help="Type of input for the classifier. Options: cls_tanh, cls_raw, poolAvg_tanh, poolAvg_raw. If split_documents is set to True, only cls_tanh is available. Default: cls_tanh.")    
    parser.add_argument("--number_classes", type=int, default=3, help="Number of classes. Default: 3")

    args = parser.parse_args()
    return(args)


def check_args(args):
    # TODO
    #if not os.path.isfile(args.classification_model):
    #    raise ValueError(".")
    if (args.input_to_classifier):
        return


if __name__ == "__main__": 
    
    args = readCommandLine()
    check_args(args)
    
    accelerator = Accelerator(dispatch_batches=False, gradient_accumulation_steps=args.gradient_accumulation_size)
    device = accelerator.device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working on" , device)    

    # TODO seed per commandline
    RANDOM_SEED = 1642
    set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    
    if(args.task == 'training'):
       trainer.trainingLoop(accelerator, device, args)
    elif (args.task == 'evaluation'):
       trainer.evaluation(device, args)
    elif (args.task == 'classification'):
       trainer.classification(device, args)
    else:
       print('Wrong task to perform: ', args.task)
    

