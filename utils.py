# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report, confusion_matrix
import trainer, data

import itertools
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
 "axes.linewidth":2,
 "xtick.major.width":2,
 "xtick.minor.width":2,
 "ytick.major.width":2,
 "ytick.minor.width":2,
 "xtick.major.size":8,
 "ytick.major.size":8,
 "xtick.minor.size":6,
 "ytick.minor.size":6
})
plt.rcParams.update({
#  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 20
})
fontTit = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'heavy',
        'size': 22,
        }
fontAx = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 18,
        }

def evaluate(model, tokenizer, dataLoader, device, args, classNames):
    y_pred, y_pred_probs, y_test = trainer.getPredictions(model, tokenizer, dataLoader, device, args)
    print(classification_report(y_test, y_pred, target_names=classNames))


def printEvalReport(y_test, y_pred, classNames):
    print(classification_report(y_test, y_pred, target_names=classNames))


def printConfusionMatrix(y_test, y_pred, classNames):
    #print(confusion_matrix(y_test, y_pred, labels=classNames))
    print(confusion_matrix(y_test, y_pred))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges):
    # modifying https://stackoverflow.com/questions/40264763/how-can-i-make-my-confusion-matrix-plot-only-1-decimal-in-python
    plt.figure(figsize=(7,6.4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=0)
    ax = plt.gca()
    ax.set_xticklabels(data.classNamesToyData(), fontsize=17)
    ax.set_yticklabels(data.classNamesToyData(), fontsize=17)
    plt.yticks(tick_marks)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.0f'), fontsize=18,
                 horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

def plotConfusionMatrix(y_test, y_pred, classNames, fileName):
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=0)
    fig, ax = plt.subplots()
    plot_confusion_matrix(cm)
    plt.tight_layout(pad=0.3)
    plt.savefig(fileName)

def printClassifications(y_pred, y_pred_probs, testName):
    outputFile = testName+'.classified'
    with open(outputFile, "w") as file:
       for pred, prob in zip(y_pred, y_pred_probs):
           file.write(str(pred.tolist()) + '\t' + str(prob.tolist()) + '\n')

