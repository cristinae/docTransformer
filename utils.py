# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report
import trainer

def evaluate(model, tokenizer, dataLoader, device, args, classNames):
    y_pred, y_pred_probs, y_test = trainer.getPredictions(model, tokenizer, dataLoader, device, args)
    print(classification_report(y_test, y_pred, target_names=classNames))
    
def printEvalReport(y_test, y_pred, classNames):
    print(classification_report(y_test, y_pred, target_names=classNames))
    
def printClassifications(y_pred, y_pred_probs, testName):
    outputFile = testName+'.classified'
    with open(outputFile, "w") as file:
       for pred, prob in zip(y_pred, y_pred_probs):
           file.write(str(pred.tolist()) + '\t' + str(prob.tolist()) + '\n')

