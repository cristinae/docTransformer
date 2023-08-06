# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F
import transformers
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed

import data
import network
import utils


def trainBatch(accelerator, model, tokenizer, documents, targets, lossFN, optimizer, device, scheduler, args):
  model = model.train()

  losses = []
  correct_predictions = 0
  nExamples = 0
  
  encodings = tokenizer(documents, truncation=args.truncation, padding=args.padding, return_token_type_ids=False, return_tensors='pt')    
#  input_ids = encodings['input_ids'].clone().detach().to(device)      # this contains sentences
#  attention_mask =  encodings["attention_mask"].clone().detach().to(device)
  input_ids = encodings['input_ids'].clone().detach()      # this contains sentences
  attention_mask =  encodings["attention_mask"].clone().detach()
  targets = targets.to(accelerator.process_index)          # this contains documents

  outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
  )

  # when the epoch finishes before a batch, we get outputs anyway (why?)
  if (len(outputs) != len(targets)):
     outputs=outputs[:len(targets)]
  
  _, preds = torch.max(outputs, dim=1)
  loss = lossFN(outputs, targets)

  nExamples += 1
  correct_predictions += torch.sum(preds == targets)
  losses.append(loss.item())

  #loss.backward()
  accelerator.backward(loss)
  nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  scheduler.step()
  optimizer.zero_grad()

  return correct_predictions.double()/nExamples/args.batch_size, np.mean(losses)


def valModel(accelerator, model, tokenizer, data_loader, lossFN, device, args):
  model = model.eval()

  losses = []
  correct_predictions = 0
  nExamples = 0

  with torch.no_grad():
    for batch in data_loader:

      if (args.split_documents):
         documents = data.split_batch(batch[0], args)
      else:
         documents = batch[0]

      encodings = tokenizer(documents, truncation=args.truncation, padding=args.padding, return_token_type_ids=False, return_tensors='pt')    
#      input_ids = encodings['input_ids'].clone().detach().to(device)
#      attention_mask =  encodings["attention_mask"].clone().detach().to(device)
      targets = batch[1].to(accelerator.process_index)
      input_ids = encodings['input_ids'].clone().detach()
      attention_mask =  encodings["attention_mask"].clone().detach()
#      targets = batch[1]

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = lossFN(outputs, targets)

      nExamples += 1
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double()/nExamples/args.batch_size, np.mean(losses)



def getPredictions(model, tokenizer, data_loader, device, args):
  model = model.eval()
  
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for batch in data_loader:

      if (args.split_documents):
         documents = data.split_batch(batch[0], args)
      else:
         documents = batch[0]

      encodings = tokenizer(documents, truncation=args.truncation, padding=args.padding, return_token_type_ids=False, return_tensors='pt')    
#      input_ids = encodings['input_ids'].clone().detach().to(device)
#      attention_mask =  encodings["attention_mask"].clone().detach().to(device)
#      targets = batch[1].to(device)
      input_ids = encodings['input_ids'].clone().detach()
      attention_mask =  encodings["attention_mask"].clone().detach()
      targets = batch[1].to(accelerator.process_index)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return predictions, prediction_probs, real_values


def getPredictionsNoTargets(model, tokenizer, data_loader, device, args):
  model = model.eval()
  
  predictions = []
  prediction_probs = []

  with torch.no_grad():
    for batch in data_loader:

      if (args.split_documents):
         documents = data.split_batch(batch[0], args)
      else:
         documents = batch[0]

      encodings = tokenizer(documents, truncation=args.truncation, padding=args.padding, return_token_type_ids=False, return_tensors='pt')    
#      input_ids = encodings['input_ids'].clone().detach().to(device)
#      attention_mask =  encodings["attention_mask"].clone().detach().to(device)
      input_ids = encodings['input_ids'].clone().detach()
      attention_mask =  encodings["attention_mask"].clone().detach()

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      predictions.extend(preds)
      prediction_probs.extend(probs)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  return predictions, prediction_probs


def trainingLoop(device, args):

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    if args.seed is not None:
        set_seed(args.seed)
    

    # Format the data for torch usage with dataloaders    
    # Creating the iterable dataset object
    trainingSet = data.DocIterableDataset(args.train_dataset)
    # Shuffling shas been deprecated
    #trainingSet = data.ShuffleDataset(trainingSet, args.buffer, args.shuffling)
    trainDataLoader = data.DataLoader(trainingSet, batch_size=args.batch_size)
    #I = 0
    #for example in trainDataLoader:
    #    accelerator.print(example)
    #    I = I+1
    #    break
    #accelerator.print(I)    
    validationSet = data.DocIterableDataset(args.validation_dataset)
    valDataLoader = data.DataLoader(validationSet, batch_size=args.batch_size)
    with open(args.train_dataset, 'r') as fp:
         for count, line in enumerate(fp):
             pass
    dataSize = count
      
    # Initialise the model
    tokenizer = network.setTokenizer(args)
    tokenizer.save_pretrained('./model/') 
    with accelerator.main_process_first():
         model = network.setModel(args, device)
    optimizer = network.setOptimizer(args, model)
    scheduler = network.setScheduler(args, optimizer, dataSize)
    lossFN = network.setLoss(device)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # not many choices here as our data are not tensors, prepare() fails otherwise
    trainDataLoader = accelerator.prepare_data_loader(trainDataLoader, device_placement=False)
    valDataLoader = accelerator.prepare_data_loader(valDataLoader, device_placement=False)

    # Training
    steps = 0
    bestAcc = 0
    trainAcc = 0
    trainLoss = 0
    modelBest = model
    for epoch in range(args.epochs):
 
       accelerator.print(f'Epoch {epoch + 1}/{args.epochs}')
       accelerator.print('-' * 10)

       n_batches = 0
       for batch in trainDataLoader:
         with accelerator.accumulate(model):
           n_batches += 1
           steps += 1
           if (args.split_documents):
              # documents contains sentences
              documents = data.split_batch(batch[0], args)
           else:
              # documents contains documents
              documents = batch[0]

           #accelerator.print('batches:', n_batches)
           trainAccBatch, trainLossBatch = trainBatch(accelerator, model, tokenizer, documents, batch[1], lossFN, optimizer, device, scheduler, args) 
           trainAcc = trainAcc + trainAccBatch
           trainLoss = trainLoss + trainLossBatch
           
           if (steps%args.eval_steps==0):
              accelerator.print(f'Step {steps}: train loss {trainLoss/args.eval_steps} accuracy {trainAcc/args.eval_steps}')
              valAcc, valLoss = valModel(accelerator,model, tokenizer, valDataLoader, lossFN, device, args)
              accelerator.print(f'              val   loss {valLoss} accuracy {valAcc}')

              if (steps%dataSize!=0):
                 trainAcc = 0
                 trainLoss = 0

              if valAcc > bestAcc:
                 accelerator.wait_for_everyone() 
                 model2save = accelerator.unwrap_model(model)
                 #torch.save(model.state_dict(), args.classification_model)
                 accelerator.save(model2save.state_dict(), args.classification_model)
                 bestAcc = valAcc
                 modelBest = model

       accelerator.print()
       accelerator.print('Steps: ', steps)
       # what's the meaning of the epoch loss?
       # print(f'Epoch {epoch+1}: train loss {trainLoss/(steps-args.eval_steps*epoch)} accuracy {trainAcc/steps-args.eval_steps*epoch}')
       # print(f'Epoch {epoch+1}: train loss {trainLoss/n_batches} accuracy {trainAcc/n_batches}')
       valAcc, valLoss = valModel(accelerator, model, tokenizer, valDataLoader, lossFN, device, args)
       accelerator.print(f'         val   loss {valLoss} accuracy {valAcc}')
       accelerator.print()
 

def evaluation(device, args):

    evalSet = data.DocIterableDataset(args.test_dataset)
    dataLoader = data.DataLoader(evalSet, batch_size=args.batch_size)
    classNames = set()
    with open(args.test_dataset, 'r') as fp:
         for line in fp:
             classNames.add(line.split('\t')[0])

    model = network.setModel(args, device)
    model.load_state_dict(torch.load(args.classification_model, map_location=torch.device(device)))
    tokenizer = network.loadTokenizer()
    y_pred, y_pred_probs, y_test = getPredictions(model, tokenizer, dataLoader, device, args)
    utils.printEvalReport(y_test, y_pred, classNames)
    

#RuntimeError: Error(s) in loading state_dict for DocTransformerClassifier:
#	size mismatch for outClasses.weight: copying a param with shape torch.Size([2, 768]) from checkpoint, the shape in current model is torch.Size([3, 768]).
def classification(device, args):

    evalSet = data.DocIterableDataset(args.test_dataset)
    dataLoader = data.DataLoader(evalSet, batch_size=args.batch_size)
    
    model = network.setModel(args, device)
    model.load_state_dict(torch.load(args.classification_model, map_location=torch.device(device)))
    tokenizer = network.loadTokenizer()
    y_pred, y_pred_probs = getPredictionsNoTargets(model, tokenizer, dataLoader, device, args)
    utils.printClassifications(y_pred, y_pred_probs,args.test_dataset) 
    
    
