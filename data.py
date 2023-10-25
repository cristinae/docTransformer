# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

import random
import numpy as np


class DocIterableDataset(IterableDataset):
    ''' Iterator on the datafile that extracts text and class'''

    def __init__(self, filename):

        self.filename = filename

    def line_mapper(self, line):
        
        columns = line.split('\t')
        if(len(columns) != 5):
            print('Format error in datafile')
            print(columns)
        label = columns[1]  #2 for politics
        label = labelVariantsData(label)
        #label = labelPoliticsData(label)
          
        doc = columns[4]  #5 for politics
        return (doc, label)

    def __iter__(self):

        file_itr = open(self.filename)
        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)
        
        return mapped_itr


''' Adapted from https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130
    Alternative to look: torch.utils.data.datapipes.iter.combinatorics.ShuffleIterDataPipe '''
class ShuffleDataset(torch.utils.data.IterableDataset):
     def __init__(self, dataset, buffer_size, shuffling):
     
      # super().__init__()
       self.dataset = dataset
       self.buffer_size = buffer_size
       self.shuffling = shuffling
  
     def __iter__(self):
       shufbuf = []
       try:
         dataset_iter = iter(self.dataset)
         for i in range(self.buffer_size):
           shufbuf.append(next(dataset_iter))
       except:
         self.buffer_size = len(shufbuf)

       if(self.shuffling):
          try:
            while True:
              try:
                item = next(dataset_iter)
                evict_idx = random.randint(0, self.buffer_size - 1)
                yield shufbuf[evict_idx]
                shufbuf[evict_idx] = item
              except StopIteration:
                break
          except StopIteration:
             return shufbuf   
       else:
          return shufbuf   
          
def get_inputIDs_separator(sep, tokenizer):
     ''' Extract the token IDs corresponding to the subunits of the sentence separator'''

     encoding = tokenizer(sep, return_token_type_ids=False, return_tensors='pt')    
     return encoding['input_ids'][0][1:-1]


def get_inputIDs_lineBreak(tokenizer):
     ''' Extract the token ID corresponding to \n'''

     encoding = tokenizer("\n", return_token_type_ids=False, return_tensors='pt')    
     return encoding['input_ids'][0][1:-1]


def split_batch(docs, args):
    ''' Splits a batch of documents using the sentence separator of the corpus. Spaces around the separator are expected.
        A line break is added between documents.'''
                
    sentence_batch = args.sentence_batch_size 
    sentences_batch = []
    for doc in docs:
        doc = doc.replace('\n', '')
        sentences_doc = doc.split(' '+args.sentence_delimiter+' ')
        # sentence_batch_size allows to fit all sentences individually
        if (len(sentences_doc) <= sentence_batch):
            sentences_batch = sentences_batch + sentences_doc + ['\n']
        # sentence_batch_size is smaller than the number of sentences in a document
        # a hand-made splitting is needed
        else:
          if(args.split_method) == 'char':
            # Naive splitting: equal number of chars, sentences (and words are broken)
            docClean = doc.replace(args.sentence_delimiter+' ', '')
            parts = [docClean[i:i+len(docClean)//sentence_batch+1] for i in range(0, len(docClean), len(docClean)//sentence_batch+1)]
            sentences_batch = sentences_batch + parts + ['\n']
          elif(args.split_method) == 'sentence':
            # Unbalanced sentence splitting            
            i = 0        #char
            part = ''    #fragment
            splitFlag = 0
            desiredLengthXFragment = len(doc)//sentence_batch
            for sent in sentences_doc:
                for char in sent:
                    part = part + char
                    i += 1
                    # This is a break point, but we wait until the end of the sentence
                    if (i%desiredLengthXFragment==0 and splitFlag==0):
                       splitFlag = 1
                # This is the end of the sentence
                if (splitFlag==1):
                    sentences_batch = sentences_batch + [part.strip(' ')] # we don't want the initial/last space
                    splitFlag = 0
                    part = ''
                # when adding the next sentence we add a space to separate it from the previous one 
                part = part + ' '  
            # We add anything that still remains and the end of document mark (\n)
            if(part!=' '):
              sentences_batch = sentences_batch + [part.strip(' ')] + ['\n']  
            else:
              sentences_batch = sentences_batch + ['\n']  
                 
            # Can we do better without exploring all the chars? We have the splitPoints and the endOfSentencePoints,
            # just a few comparisons should be enough
            #parts = []
            #splitPoints = [i*desiredLengthXFragment for i in range(1,sentence_batch)]
            #ngram = doc[splitPoints[0] : splitPoints[0]+6]
            #for i in splitPoints:
            #    k = 
            #    while (ngram != ' '+args.sentence_delimiter+' '):

    return sentences_batch


def classNamesToyData(): 
    return ['Chile', 'España', 'México', 'Mix']
    
def labelVariantsData(label):
    d = {'cl':0, 'es':1, 'mx':2, 'mix':3}
    return d[label]
 
def classNamesPoliticsData(): 
    return ['Left', 'Right']
    
def labelPoliticsData(label):
    d = {'left':0, 'right':1}
    return d[label]
 
