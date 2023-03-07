# docTransformer
Transformer for classification tasks that operates with document fragments


## Features

* UPCOMING: gradient acumulation
* UPCOMING: pert distribution for building document embeddings
* Multi-node, multi-GPU implementation using the ```Accelerate``` library
* An input data streaming implementation allows training with large datasets
* Possibility to build document embeddings before classification both during training and classification
* Several possibilities to be considered as input to the classifier besides the ```[CLS]``` token


## Requirements

* [PyTorch](http://pytorch.org/) version >= 1.9.1

