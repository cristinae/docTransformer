# -*- coding: utf-8 -*-

import transformers
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
import torch
from torch import nn, optim


class DocTransformerClassifier(nn.Module):
  '''Network definition '''

  def __init__(self, nClasses, args, device):
    super(DocTransformerClassifier, self).__init__()
    # add_pooling_layer=False is needed for parallelism
    # https://github.com/UKPLab/sentence-transformers/issues/1454
    # https://github.com/UKPLab/sentence-transformers/pull/1215
    self.transformer = RobertaModel.from_pretrained(args.pretrained_model, add_pooling_layer=False, return_dict=False)
    self.device = device
    self.batch_size = args.batch_size
    self.split_docs = args.split_documents
    self.pooling_method = args.input_to_classifier
    self.dropPre = nn.Dropout(p=args.dropout_prepooling)
    self.densePre =  nn.Linear(self.transformer.config.hidden_size, self.transformer.config.hidden_size)
    self.dropPost = nn.Dropout(p=args.dropout_postpooling)
    self.outClasses = nn.Linear(self.transformer.config.hidden_size, nClasses)
    
    # Fine-tune the model or not while learning the classifier
    if (args.freeze_pretrained):
       self.freeze_pretrained()
    else:
       self.unfreeze_pretrained()  
       
  

  def forward(self, input_ids, attention_mask):
    ''' Forward pass with several pooling methods implemented as input for the classifier in the sentence level setting
        and fragment splitting for fragment average as input for the classifier in the document level setting.'''
  
    last_hidden_state, pooled_output = self.transformer(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    # last_hidden_state shape (batch_size, sequence_length, hidden_size)) Sequence of hidden-states at the output of the last layer of the model.
    torch.set_printoptions(threshold=10000)
   
    
    #Split de documents into sentences before the final pooling
    if (self.split_docs):   
       lineBreak = [0, 203, 2] # Achtung! Hardcoded for Roberta tokeniser (but we don't have the tokeniser at this point)
       last_hidden_batch = torch.zeros([1, self.transformer.config.hidden_size], dtype=torch.float32).to(self.device)
       last_hidden_state_average = torch.zeros([self.batch_size, self.transformer.config.hidden_size], dtype=torch.float32).to(self.device)
       #last_hidden_batch = torch.zeros([1, self.transformer.config.hidden_size], dtype=torch.float32)
       #last_hidden_state_average = torch.zeros([self.batch_size, self.transformer.config.hidden_size], dtype=torch.float32)
       batch = 0
       sentences_in_batch = 0       
       for sentence_ids, last_hidden in zip(input_ids, last_hidden_state):
           last_hidden_batch = last_hidden_batch.add(last_hidden[0,:])
           sentences_in_batch += 1
           if (sentence_ids[0:3].tolist()==lineBreak):   
              last_hidden_state_average[batch] = torch.div(last_hidden_batch, sentences_in_batch)
              batch += 1
              sentences_in_batch = 0
              last_hidden_batch = torch.zeros([1, self.transformer.config.hidden_size], dtype=torch.float64).to(self.device)
              #last_hidden_batch = torch.zeros([1, self.transformer.config.hidden_size], dtype=torch.float64)
       # This is the equivalent to cls_tanh
       pooled_bymethod = self.tanhPrep(last_hidden_state_average)
    else: 
       # Different pooling methods. Supported options: cls_tanh, cls_raw, poolAvg_tanh, poolAvg_raw
       if (self.pooling_method == 'cls_raw'):
          pooled_bymethod = last_hidden_state[:, 0, :]
       elif (self.pooling_method == 'cls_tanh'):
          pooled_bymethod = last_hidden_state[:, 0, :]
          pooled_bymethod = self.tanhPrep(pooled_bymethod)
       elif (self.pooling_method == 'poolAvg_raw'):
          pooled_bymethod = mean_pool(last_hidden_state, attention_mask)
       elif (self.pooling_method == 'poolAvg_tanh'):
          pooled_bymethod = mean_pool(last_hidden_state, attention_mask)
          pooled_bymethod = self.tanhPrep(pooled_bymethod)
       else:
          pooled_bymethod = pooled_output
          print("Careful!", self.pooling_method)
    
    output = self.dropPost(pooled_bymethod)
    return self.outClasses(output)
            
  def tanhPrep(self, pooled_bymethod):
       pooled_bymethod = self.dropPre(pooled_bymethod)
       pooled_bymethod = self.densePre(pooled_bymethod)
       return torch.tanh(pooled_bymethod)

  def freeze_pretrained(self):
    for param in self.transformer.named_parameters():
        param[1].requires_grad=False

  def unfreeze_pretrained(self):
    for param in self.transformer.named_parameters():
        param[1].requires_grad=True



def mean_pool(token_embeds, attention_mask):
    '''Mean pooling definition for the non-PAD tokens (copied)'''

    # reshape attention_mask to cover dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask): sum_embeddings / sum_mask
    poolAvg = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)

    return poolAvg
        
        
def setTokenizer(args):
   
   return RobertaTokenizer.from_pretrained(args.pretrained_model)

def loadTokenizer():
   
   return RobertaTokenizer.from_pretrained('./model/')

def setModel(args, device, nClasses):

   model = DocTransformerClassifier(nClasses, args, device)
   model = model.to(device)  # adding accelerate
   
   return(model)


def setOptimizer(args, model):

   optimizer = optim.AdamW(model.parameters(), args.lr)
   return(optimizer)
   
   
def setScheduler(args, optimizer, dataSize):

   totalSteps = dataSize*args.epochs
   scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=totalSteps)

   return(scheduler)


def setLoss(device):

   lossFN = nn.CrossEntropyLoss().to(device) # adding accelerate
   #lossFN = nn.CrossEntropyLoss()
   return(lossFN)

   
