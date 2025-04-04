# SymRep :  A Systematic Evaluation of Symbolic Music Representations

This repo contains the code for the paper:
[Symbolic Music Representations for Classification Tasks: A Systematic Evaluation]()

It contains the models and experiments for evaluating three symbolic music representations:
- Matrix-like (Pianoroll)
- Graph-like 
- Sequence-like (Using Tokenized representations mainly from MidiTok)

We use two datasets for evaluation:
- [ASAP](https://github.com/CPJKU/asap-dataset)
- [ATEPP](https://github.com/BetsyTang/ATEPP)

For their dataset split used in the experiment, please refer to the refined metadata files: [Google Drive](https://drive.google.com/drive/folders/1vyByrctZX9hJ90pz0vtrqLmZ2XPykCIH?usp=drive_link)

To use please download the corresponding datasets and refer to the path in ```conf/config.yaml```

#### Requirements
- Python 3.6 or newer

To install all python dependencies, run:
```bash
pip install -r requirements.txt
```

#### Usage

To run the experiments, run the following command and change the parameters:
```bash
./experiments/crossval_run.sh
```

#### Configuration 

The configurations regarding experiment, task, dataset, as well as the three representations, can be modified in  ```conf/config.yaml```. The parameters management are supported by Hydra.

Experiments for this repository are logged using [Weights and Biases](https://wandb.ai/huanz/symrep).

#### Structure

The code is organized into the following folders:
- `conf`: configuration files
- `model`: contains the code for the models used in the paper
- `experiments`: contains the code for the experiments in the paper
- `converters`: contains the code for converting each symbolic representation
    - `miditok`:  contains the custom tokenizers for musicxml  
