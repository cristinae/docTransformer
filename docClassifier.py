# -*- coding: utf-8 -*-

import warnings
import argparse
import os

import torch
import trainer

import numpy as np
import random

    
def readCommandLine():
    ''' Parser for command line arguments'''
    
    def boolean_string(s):
        if s not in {'False', 'True'}:
           raise ValueError('Not a valid boolean string')
        return s == 'True'
        
    parser = argparse.ArgumentParser(description="Fine-tuning Roberta for document classification")

    # Input/Output
    parser.add_argument("-c", "--train_dataset", required=False, type=str, default='./data/multivariant3all.train', help="Training dataset for classification")
    parser.add_argument("-v", "--validation_dataset", type=str, default='./data/multivariant3all.dev', help="Validation set")
    parser.add_argument("-t", "--test_dataset", type=str, default='./data/multivariant3all.test', help="Test set to evaluate or classify")
    parser.add_argument("-o", "--classification_model", required=False, type=str, default='modelBest.bin', help="Name for the model file. Default: modelBest.bin")
    parser.add_argument("-f", "--classification_model_folder", required=False, type=str, default='./model/', help="Name for the model folder (it will also contain checkpoints). Default: ./model")
    # Shuffling has been deprecated, please shuffle the training data beforehand
    #parser.add_argument("--buffer", type=int, default=1, help="Documents are not loaded completely into memory but into <buffer> chunks. Default: 100000 documents.")
    #parser.add_argument("--shuffling", type=boolean_string, default=True, help="Suffling within a dataset buffer. Options: True, False. Default: True.")
    # Checkpointing
    parser.add_argument("--resume_from_checkpoint", type=str, default='False', help="If the training should continue from a checkpoint folder give the folder name, otherwise False. Default: False")
    parser.add_argument("--num_checkpoints", type=int, default=1, help="Number of checkpoints to keep. Default: 1")

    #Task (training by default)
    parser.add_argument("--task", type=str, default='training', help="Task to perform. Options: training, evaluation, classification. Default: training")
    parser.add_argument("--plotConfusionFileName", type=str, default='confusionTest.png', help="File name for the plot of the confusion matrix when task evaluation is chosen. Default: confusionTest.png")
    
    # Data processing
    # Tokenisation
    parser.add_argument("--truncation", type=boolean_string, default=True, help="")
    parser.add_argument("--padding", type=boolean_string, default=True, help="")
    parser.add_argument("--max_length", type=int, default=None, help="")
    
    parser.add_argument("--split_documents", type=boolean_string, default=False, help="Deal with documents as a set of fragments. Default: False")
    parser.add_argument("--sentence_delimiter", type=str, default='<NS>', help="Delimiter used to separate fragments in a document. Default: <NS>")
    parser.add_argument("--split_method", type=str, default='sentence', help="How to split the document. Options: char (exact splitting at char level), sentence (aprox splitting at sentence level). Default: sentence.")    

    # Base model
    # spanish model skimai/spanberta-base-cased; german model 
    # xlm-roberta-large
    parser.add_argument("-m", "--pretrained_model", type=str, default='xlm-roberta-large', help="pretrained model (currently only Roberta family implemented)")
    parser.add_argument("--freeze_pretrained", type=boolean_string, default=False, help="Freeze weights of the pretrained model. Options: True, False")
    
    # Training
    parser.add_argument("-e", "--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Number of documents in a batch. Default: 8")
    parser.add_argument("--sentence_batch_size", type=int, default=20, help="Number of sentences per document in a batch. Default: 20.")
    parser.add_argument("--eval_steps", type=int, default=8000, help="Number of batches prior to validation. Default: 8000")

    parser.add_argument("-a", "--gradient_accumulation_steps", type=int, default=1, help="Number of steps before updating the gradients")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for AdamW. Default: 2e-5 (5e-6 paper).")
    
    parser.add_argument("--dropout_prepooling", type=float, default=0.1, help="Dropout to be applied after retrieving the embeddings and before pooling. Default: 0.1.")

    # Classifier
    parser.add_argument("-d", "--dropout_postpooling", type=float, default=0.1, help="Dropout to be applied before the last linear layer of the classifier. Default: 0.1.")
    parser.add_argument("--input_to_classifier", type=str, default='cls_tanh', help="Type of input for the classifier. Options: cls_tanh, cls_raw, poolAvg_tanh, poolAvg_raw. If split_documents is set to True, only cls_tanh is available. Default: cls_tanh.")    
    parser.add_argument("--number_classes", type=int, default=4, help="Number of classes. Default: 4")

    # Utils
    parser.add_argument("--seed", type=int, default=1642, help="Seed to be used by torch, numpy and random (int)")

    args = parser.parse_args()
    return(args)


def check_args(args):
    # TODO
    if (args.input_classifier):
        return


if __name__ == "__main__": 
    
    args = readCommandLine()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working on" , device)    

        
    # the trained model will be stored in 'model'
    if not os.path.exists('model'):
       os.makedirs('model')
    

    if(args.task == 'training'):
       trainer.trainingLoop(device, args)
    elif (args.task == 'evaluation'):
       trainer.evaluation(device, args)
    elif (args.task == 'classification'):
       trainer.classification(device, args)
    else:
       print('Wrong task to perform: ', args.task)
    

