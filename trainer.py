# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F
import transformers
import numpy as np
import string

from captum.attr import IntegratedGradients, LayerIntegratedGradients, TokenReferenceBase
from scipy.signal import argrelextrema
from collections import Counter

from accelerate import Accelerator

import data
import network
import utils
import shutil
import os


def trainBatch(accelerator, model, tokenizer, documents, targets, lossFN, optimizer, device, scheduler, args):
  model = model.train()

  losses = []
  correct_predictions = 0
  nExamples = 0
  
  encodings = tokenizer(documents, truncation=args.truncation, padding=args.padding, return_token_type_ids=False, return_tensors='pt')    
  input_ids = encodings['input_ids'].clone().detach().to(device)      # this contains sentences
  attention_mask =  encodings["attention_mask"].clone().detach().to(device)
#  input_ids = encodings['input_ids'].clone().detach()      # this contains sentences
#  attention_mask =  encodings["attention_mask"].clone().detach()
  targets = targets.to(device)                                        # this contains documents

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


def valModel(model, tokenizer, data_loader, lossFN, device, args):
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
      input_ids = encodings['input_ids'].clone().detach().to(device)
      attention_mask =  encodings["attention_mask"].clone().detach().to(device)
      targets = batch[1].to(device)
#      input_ids = encodings['input_ids'].clone().detach()
#      attention_mask =  encodings["attention_mask"].clone().detach()
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
      input_ids = encodings['input_ids'].clone().detach().to(device)
      attention_mask =  encodings["attention_mask"].clone().detach().to(device)
      targets = batch[1].to(device)
#      input_ids = encodings['input_ids'].clone().detach()
#      attention_mask =  encodings["attention_mask"].clone().detach()
#      targets = batch[1]

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
      input_ids = encodings['input_ids'].clone().detach().to(device)
      attention_mask =  encodings["attention_mask"].clone().detach().to(device)
#      input_ids = encodings['input_ids'].clone().detach()
#      attention_mask =  encodings["attention_mask"].clone().detach()

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

    # Format the data for torch usage with dataloaders    
    # Creating the iterable dataset object
    trainingSet = data.DocIterableDataset(args.train_dataset, args.task)
    # Shuffling shas been deprecated
    #trainingSet = data.ShuffleDataset(trainingSet, args.buffer, args.shuffling)
    trainDataLoader = data.DataLoader(trainingSet, batch_size=args.batch_size)
    #I = 0
    #for example in trainDataLoader:
    #    accelerator.print(example)
    #    I = I+1
    #    break
    #accelerator.print(I)    
    validationSet = data.DocIterableDataset(args.validation_dataset, args.task)
    valDataLoader = data.DataLoader(validationSet, batch_size=args.batch_size)
    with open(args.train_dataset, 'r') as fp:
         for count, line in enumerate(fp):
             pass
    dataSize = count
      
    # Initialise the model
    tokenizer = network.setTokenizer(args)
    tokenizer.save_pretrained('./model/')  
    model = network.setModel(args, device)
    optimizer = network.setOptimizer(args, model)
    scheduler = network.setScheduler(args, optimizer, dataSize)
    lossFN = network.setLoss(device)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # not many choices here as our data are not tensors, prepare() fails otherwise
    trainDataLoader = accelerator.prepare_data_loader(trainDataLoader, device_placement=False)
    valDataLoader = accelerator.prepare_data_loader(valDataLoader, device_placement=False)

    startingEpoch = 0
    overallSteps = 0
    resumeStep = None
    # are we resuming from a previous training?
    if args.resume_from_checkpoint !="False" and args.resume_from_checkpoint != "":
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        # Extract `step_{i}`
        training_difference = os.path.splitext(path)[0]
        training_difference = training_difference.rstrip('/')
        resumeStep = int(training_difference.replace("step_", ""))
        startingEpoch = resumeStep // dataSize
        resumeStep -= startingEpoch * dataSize


    # Training
    steps = 0
    bestAcc = 0
    trainAcc = 0
    trainLoss = 0
    modelBest = model
    for epoch in range(startingEpoch, args.epochs):
 
       accelerator.print(f'Epoch {epoch + 1}/{args.epochs}')
       accelerator.print('-' * 10)

       if args.resume_from_checkpoint and epoch == startingEpoch and resumeStep is not None:
          # We need to skip steps until we reach the resumed step
          activeDataLoader = accelerator.skip_first_batches(trainDataLoader, resumeStep)
          overallSteps += resumeStep
       else:
          # After the first iteration though, we need to go back to the original dataloader
          activeDataLoader = trainDataLoader

       n_batches = 0
       for batch in activeDataLoader:
         with accelerator.accumulate(model):
           n_batches += 1
           steps += 1
           overallSteps += 1
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
           
           if (overallSteps%args.eval_steps==0):
              # Save a checkpoint everytime we validate
              if (args.num_checkpoints>0):
                  #first we remove the previous checkpoint, the last we do not want to keep
                  chkToDelete = overallSteps-args.num_checkpoints*args.eval_steps
                  outputToDelete = args.classification_model_folder+'/step_'+str(chkToDelete)
                  if os.path.isdir(outputToDelete):
                      shutil.rmtree(outputToDelete)
                  # then we save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                  outputCHK = f"step_{overallSteps}"
                  outputCHK = os.path.join(args.classification_model_folder, outputCHK)
                  accelerator.save_state(outputCHK)
              # Validation
              accelerator.print(f'Step {overallSteps}: train loss {trainLoss/args.eval_steps} accuracy {trainAcc/args.eval_steps}')
              valAcc, valLoss = valModel(model, tokenizer, valDataLoader, lossFN, device, args)
              accelerator.print(f'              val   loss {valLoss} accuracy {valAcc}')

              if (overallSteps%dataSize!=0):
                 trainAcc = 0
                 trainLoss = 0

              if valAcc > bestAcc:
                 model2save = accelerator.unwrap_model(model)
                 #torch.save(model.state_dict(), args.classification_model)
                 accelerator.save(model2save.state_dict(), os.path.join(args.classification_model_folder, args.classification_model))
                 bestAcc = valAcc
                 modelBest = model

       accelerator.print()
       # what's the meaning of the epoch loss?
       # print(f'Epoch {epoch+1}: train loss {trainLoss/(steps-args.eval_steps*epoch)} accuracy {trainAcc/steps-args.eval_steps*epoch}')
       # print(f'Epoch {epoch+1}: train loss {trainLoss/n_batches} accuracy {trainAcc/n_batches}')
       valAcc, valLoss = valModel(model, tokenizer, valDataLoader, lossFN, device, args)
       accelerator.print(f'         val   loss {valLoss} accuracy {valAcc}')
       accelerator.print()
 

def evaluation(device, args):

    evalSet = data.DocIterableDataset(args.test_dataset, args.task)
    dataLoader = data.DataLoader(evalSet, batch_size=args.batch_size)
    classNames = set()
    with open(args.test_dataset, 'r') as fp:
         for line in fp:
             classNames.add(line.split('\t')[1])

    model = network.setModel(args, device)
    model.load_state_dict(torch.load(os.path.join(args.classification_model_folder, args.classification_model), map_location=torch.device(device)))
    tokenizer = network.loadTokenizer()
    y_pred, y_pred_probs, y_test = getPredictions(model, tokenizer, dataLoader, device, args)
    utils.printEvalReport(y_test, y_pred, classNames)
    utils.printConfusionMatrix(y_test, y_pred, list(classNames))
    utils.plotConfusionMatrix(y_test, y_pred, list(classNames), args.plot_confusion_fileName)
    

def classification(device, args):

    evalSet = data.DocIterableDataset(args.test_dataset, args.task)
    dataLoader = data.DataLoader(evalSet, batch_size=args.batch_size)
    
    model = network.setModel(args, device)
    model.load_state_dict(torch.load(os.path.join(args.classification_model_folder, args.classification_model), map_location=torch.device(device)))
    tokenizer = network.loadTokenizer()
    y_pred, y_pred_probs = getPredictionsNoTargets(model, tokenizer, dataLoader, device, args)
    modelName = args.classification_model[:-4] #removing the extension
    utils.printClassifications(y_pred, y_pred_probs,args.test_dataset,modelName) 


def getInput4XAI(model, tokenizer, data_loader, device, args):
    model = model.eval()
  
    docs = []
    ids = []
    attmasks = []
    tgts = []

    with torch.no_grad():
      for batch in data_loader:

        if (args.split_documents):
           documents = data.split_batch(batch[0], args)
        else:
           documents = batch[0]

        encodings = tokenizer(documents, truncation=args.truncation, padding=args.padding, return_token_type_ids=False, return_tensors='pt')    
        input_ids = encodings['input_ids'].clone().detach().to(device)
        attention_mask =  encodings["attention_mask"].clone().detach().to(device)
        targets = batch[1].to(device)
#       input_ids = encodings['input_ids'].clone().detach()
#       attention_mask =  encodings["attention_mask"].clone().detach()
#       targets = batch[1]

        docs.append(documents)
        tgts.append(targets)
        ids.append(input_ids)
        attmasks.append(attention_mask)

    return docs, tgts, ids, attmasks


def explanation(device, args):

    evalSet = data.DocIterableDataset(args.test_dataset, args.task)
    dataLoader = data.DataLoader(evalSet, batch_size=args.batch_size)
    
    model = network.setModel(args, device)
    model.load_state_dict(torch.load(os.path.join(args.classification_model_folder, args.classification_model), map_location=torch.device(device)))
    tokenizer = network.loadTokenizer()
    docs, tgts, ids, attmasks = getInput4XAI(model, tokenizer, dataLoader, device, args)

    # prepare captum
    lig = LayerIntegratedGradients(model, model.transformer.embeddings)
    token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)

    labels = data.getClassesLabelVariant()
    for label in labels:
        globals()[f"top_Positive_{label}"] = Counter()
        globals()[f"top_Negative_{label}"] = Counter() 
    
    for doc_num, doc_text in enumerate(docs):

        label = data.getTextLabel(tgts[doc_num][0].item())
        # calculate attributes
        reference_indices = token_reference.generate_reference(sequence_length=ids[doc_num].size(dim=1), device=device).unsqueeze(0)
        attr, delta = lig.attribute(inputs=ids[doc_num], baselines=reference_indices, additional_forward_args=(attmasks[doc_num]), return_convergence_delta=True, n_steps=50, internal_batch_size=args.xai_lig_batch, target=tgts[doc_num])
        # torch.Size([1, 203, 1024])
        # summarize attributions for each word token in the sequence
        attr = attr.sum(dim=-1).squeeze(0)
        attr = attr / torch.norm(attr)
    
        # convert the subunit-based output to words and afterwards build phrases 
        attribXid = dict()
        if (device=='cpu'):
            attribs=attr[doc_num].numpy() #works, but why?
        elif (device=='cuda'):
            attribs=attr.cpu().numpy()
        else:
            attribs=attr.cpu().numpy() #waiting to see tups

        for elem, id in zip(attribs,ids[doc_num][0].cpu().numpy()):   #what would be the best practice for this?
            attribXid[id]=elem        
        document = doc_text[0].replace('<NS>','') #this should not happen in the splitted setting
        document = document.translate(str.maketrans('', '', string.punctuation))
        document = document.translate(str.maketrans('', '','”“¿'))
        final_attribs = []
        words_doc = document.split()
        for word in words_doc:
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            attribution = 0
            # the attribution of a word is the sum of the attribution for its subunits
            for id in word_ids:
                if id in attribXid:
                   attribution = attribution + attribXid[id]
            final_attribs.append(attribution)
            #print(word,attribution)
        tmp = np.array(final_attribs, dtype='float32')
        # we look for local maxima and all the points around that are within the threshold. 
        # That is a phrase. We limit the length of a phrase to 5: head+-2
        threshold_attr = np.percentile(tmp[tmp!=0], args.xai_threshold_percentile)
        threshold_attr_phrase = np.percentile(tmp[tmp!=0], args.xai_threshold_percentile-10)
        #print(threshold_attr, "\n\n\n")
        local_max = argrelextrema(tmp, np.greater)
        for point in local_max[0]:
            if (final_attribs[point] > threshold_attr):
                candidate = words_doc[point]
                #candidate = words_doc[point]+'_head'
                if (len(final_attribs) > point+1 and final_attribs[point+1] > threshold_attr_phrase): 
                    candidate = candidate + ' ' + words_doc[point+1]   
                    if (len(final_attribs) > point+2 and final_attribs[point+2] > threshold_attr_phrase): 
                        candidate = candidate + ' ' + words_doc[point+2]   
                if (point > 0 and final_attribs[point-1] > threshold_attr_phrase): 
                    candidate = words_doc[point-1] + ' ' + candidate   
                    if (point > 1 and final_attribs[point-2] > threshold_attr_phrase): 
                        candidate = words_doc[point-2] + ' ' + candidate   
                globals()[f'top_Positive_{label}'].update([candidate])
    
                
    print('Top-',args.xai_elements, ' phrases according to the IG attribution score')
    for label in labels:
        print(label, end=",")
    print()
    for elem in range(0,args.xai_elements):
        for label in labels:
            if (len(globals()[f"top_Positive_{label}"].most_common(args.xai_elements))>elem):
                print(globals()[f"top_Positive_{label}"].most_common(args.xai_elements)[elem], end=",")
            else:
                print("('--',0)", end=",")
        print()

