# docTransformer
Transformer for classification tasks that operates with document fragments


## Features

* UPCOMING: pert distribution for building document embeddings
* Gradient accumulation to train with larger effective batches using the ```Accelerate``` library
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

* [Python](https://www.python.org) version >= 3.9
* [PyTorch](http://pytorch.org/) version >= 2.0.1
* [Accelerate](https://github.com/huggingface/accelerate) version >= 0.21.0

## Example Usage

### Slurm 

``` srun --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --gradient_accumulation_size 2```
