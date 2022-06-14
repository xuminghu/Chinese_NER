# CrossNER

## Installation

The code is based on PyTorch 1.7.0 and Python 3.7.7. For training, a GPU is recommended to accelerate the training speed.

### Dependencies

The code is based on Python 3.7. Its dependencies are summarized in the file `requirements.txt`.

```
numpy==1.20.1
torch==1.7.0
tqdm==4.62.3
transformers==4.19.2
```

You can install these dependencies like this:

```
pip3 install -r requirements.txt
```

## Usage

### Training

- Pretrain the full model on Conll2003 dataset first, then train it on specific target domain with default hyperparameter settings . Use `target` domain as an example.    

       `python3 main.py --num_source_tag 17 --batch_size 16 --tgt_dm science`  

        `num_source_tag` should equal to the number of classes of source domain, `num_target_tag` should equal to the number of classes of target domain and `--src_dm` is the source domain name, `--tgt_dm` is the target domain name.

### Data

You need to declare all your domain labels at first in ./src/dataloader.py line 29.
Each dataset is a folder under the `./ner_data` folder. You must name your folder as its domain name, which must be the same as the `--src_dm`, `--tgt_dm` parameter:

```
./ner_data
└── source
    ├── train
    ├── dev
    ├── test
```


