# docTransformer
Transformer for classification tasks that operates with document fragments


## Features

* UPCOMING: Multi-GPU support using the ```Accelerate``` library
* Explainability measures
  * Currently output salient words and phrases in the classification using Layer Integrated Gradients attribution scores with the ```Captum``` library
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
* [Captum](https://captum.ai/) version 0.6.0


## Example Usage

### Slurm 

#### Training

```srun --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --gradient_accumulation_steps 2```

```srun -p A100-80GB -t 3-0 --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --eval_steps 10000 --lr 5e-6 -f modelb2a8sentence16A80 -o modelb2a8sentence16A80seed2.bin -b2 -a8 --sentence_batch_size 16 --split_documents True  --seed 5678``` 

#### Evaluation

```srun -p RTXA6000  --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --task evaluation -f modelb2a8sentence2V100 -o modelb2a8sentence2V100seed3_333.bin -b2 --sentence_batch_size 2 --split_documents True --test_dataset data/multivariant3all.test --plotConfusionFileName modelSplit2Seed3test.png```

#### Classification

```srun --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --task classification -f modelb2a8sentence2V100 -o modelb2a8sentence2V100seed3_333.bin -b1 --sentence_batch_size 2 --split_documents True --test_dataset ../es/es_meta_part_1.jsonl.unk```

#### Explanation

```srun -p RTXA6000  --ntasks 1 --gpus-per-task 1 python -u docClassifier.py --task explanation -t data/testExample.mx -f modelb2a8fixV100 -o modelb2a8fixV100seed3_3.bin -b1 --split_documents False  --xai_threshold_percentile 90```


## Further Usage

In order to use it for your own classification task with full functionalities:
* Prepare your training data with the class in the 2nd column and the text to classify in the 5th column. Otherwise, modify the ```line_mapper``` function in ```data.py```
* Adapt the classification labels in the last functions of  ```data.py```


## Citation

Please, use the following bibtex entries when citing this research work

```
@inproceedings{espana-bonet-barron-cedeno-2024-elote,
    title = "Elote, Choclo and Mazorca: on the Varieties of {S}panish",
    author = "Espa{\~n}a-Bonet, Cristina  and
      Barr{\'o}n-Cede{\~n}o, Alberto",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.204",
    pages = "3689--3711"
}
```

Version v1.0.1 without the document level functionality (```--split_documents False```) has been used in

```
@InProceedings{espana-bonet-2023-multilingual,
    title = "Multilingual Coarse Political Stance Classification of Media. The Editorial Line of a {C}hat{GPT} and {B}ard Newspaper",
    author = "Espa{\~n}a-Bonet, Cristina",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.787",
    doi = "10.18653/v1/2023.findings-emnlp.787",
    pages = "11757--11777"
}
```

