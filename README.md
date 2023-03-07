# docTransformer
Transformer for classification tasks that operates with document fragments


## Features

* UPCOMING: pert distribution for building document embeddings
* Multi-node and multi-GPU support using the ```Accelerate``` library
* Gradient accumulation to train with larger effective batches
* An input data streaming implementation to allow training with large datasets
* Possibility to build document embeddings before classification both during training and classification
  * Document embedding built as the average of the ```[CLS]``` token of _n_ parts of the document:
     - the document is divided in _n_ parts with an equal number of characters. No sentence/fragment information is used
     - the document is divided in _n_ parts with an approximate equal number of sentences. If the *sentence_batch* is larger than _n_, sentences are averaged individually 
* Several possibilities to be considered as input to the classifier besides the standard ```[CLS]``` token
  * [CLS] + tanh
  * [CLS]
  * average pooling + tanh 
  * average pooling
 

## Requirements

* [PyTorch](http://pytorch.org/) version >= 1.9.1

## Example Usage

### Slurm 

``` srun --ntasks 1 --gpus-per-task 4  accelerate launch --multi_gpu docClassifier.py --gradient_accumulation_size 2```
