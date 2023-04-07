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
To use please download the corresponding datasets and place them in the `data` folder.

#### Requirements
- Python 3.6 or newer

To install all python dependencies, run:
```bash
pip install -r requirements.txt
```

#### Usage

To run the experiments, run the following command:
```bash
./experiments/run.sh
```

Experiments for this repository are logged using [Weights and Biases](wandb.ai).

#### Structure

The code is organized into the following folders:
- `model`: contains the code for the models used in the paper
- `experiments`: contains the code for the experiments in the paper
- `converters`: contains the code for converting each symbolic representation



##### Data:
After dropping the tail from data. 

| ASAP composer (perf&score) |     | ATEPP composer (perf) |      | ATEPP composer (score) |      | ATEPP performer |      |  ASAP difficulty  |       |
|----------------------------|-----|-----------------------|------|------------------------|------|-----------------|------|-------------------|-------|
| Beethoven                  | 195 | Beethoven             | 3523 | Beethoven              | 3033 | Richter         | 1581 |  9                |  164  |
| Bach                       | 163 | Chopin                | 1826 | Chopin                 | 1739 | Ashkenazy       | 1188 |  8                |  176  |
| Chopin                     | 162 | Bach                  | 1313 | Mozart                 | 653  | Arrau           | 833  |  7                |  132  |  
| Liszt                      | 67  | Schumann              | 1176 | Schubert               | 264  | Brendel         | 743  |  6                |  150  |
| Schubert                   | 55  | Schubert              | 781  | Debussy                | 254  | Kempff          | 609  |  5                |  56   |
| Schumann                   | 26  | Mozart                | 714  | Schumann               | 243  | Barenboim       | 603  |  4                |  23   |
| Haydn                      | 23  | Debussy               | 321  | Bach                   | 231  | Schiff          | 595  |                   |       |
| Mozart                     | 10  | Rach                  | 216  | Ravel                  | 169  | Horowitz        | 576  |                   |       |
| Scriabin                   | 9   | Ravel                 | 255  | Liszt                  | 122  | Gulda           | 459  |                   |       |
| Ravel                      | 9   | Prokofiev             | 241  |                        |      | Gieseking       | 362  |                   |       |
|                            |     | Liszt                 | 207  |                        |      | Gould           | 326  |                   |       |
|                            |     | Brahm                 | 199  |                        |      | Gilels          | 322  |                   |       |
|                            |     | Scriabin              | 168  |                        |      | Perahia         | 288  |                   |       |
|                            |     | Tchaikovsky           | 106  |                        |      | Pollini         | 256  |                   |       |
|                            |     | Shostakovich          | 101  |                        |      | Argerich        | 240  |                   |       |
|                            |     |                       |      |                        |      | Schnabel        | 240  |                   |       |
|                            |     |                       |      |                        |      | François        | 234  |                   |       |
|                            |     |                       |      |                        |      | Uchida          | 210  |                   |       |
|                            |     |                       |      |                        |      | Casadesus       | 164  |                   |       |
|                            |     |                       |      |                        |      | Lugansky        | 125  |                   |       |
|                            |     |                       |      |                        |      | Cortot          | 124  |                   |       |
|                            |     |                       |      |                        |      | Lang            | 115  |                   |       |
|                            |     |                       |      |                        |      | Larrocha        | 110  |                   |       |
|                            |     |                       |      |                        |      | Sokolov         | 106  |                   |       |
|                            |     |                       |      |                        |      | Lupu            | 104  |                   |       |