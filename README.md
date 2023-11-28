# docTransformer
Transformer for classification tasks that operates with document fragments


## Features

* UPCOMING: Multi-GPU support using the ```Accelerate``` library
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

```srun --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --gradient_accumulation_steps 2```

```srun -p A100-80GB -t 3-0 --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --eval_steps 10000 --lr 5e-6 -f modelb2a8sentence16A80 -o modelb2a8sentence16A80seed2.bin -b2 -a8 --sentence_batch_size 16 --split_documents True  --seed 5678``` 

```srun -p V100-16GB --ntasks 1 --gpus-per-task 4 accelerate launch --multi_gpu --num_processes 4 --num_machines 1 docClassifier.py -b1 -a2 -o best_model_multi4gpus.bin```

```srun --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --task classification -f modelb2a8sentence2V100 -o modelb2a8sentence2V100seed3_333.bin -b1 --sentence_batch_size 2 --split_documents True --test_dataset ../es/es_meta_part_1.jsonl.unk```

## Citation

Version v1.0.1 without the document level functionality (```--split_documents False```) has been used in


```
@inproceedings{espana-bonet-2023-multilingual,
    title = "Multilingual Coarse Political Stance Classification of Media. The Editorial Line of a ChatGPT and Bard Newspaper",
    author = "Espa{\~n}a-Bonet, Cristina",
    booktitle = "Findings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/",
    pages = "--"
}
```
 

