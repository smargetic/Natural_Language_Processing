# English → Italian Machine Translation (PyTorch)

A neural machine translation project built from scratch in PyTorch. The RNN and LSTM cells, attention, the Transformer, beam search, byte-pair encoding and BLEU scoring are all written by hand, so every step can be read and followed.

It's trained on English–Italian sentence pairs from [Tatoeba](https://tatoeba.org) and translates **English into Italian**.

---

## Start here

| If you want to… | Open |
|---|---|
| understand how the model works | [`Translation_Model_Pytorch.ipynb`](Translation_Model_Pytorch.ipynb): every part explained, with the code. Runs in seconds and trains nothing. |
| train models, compare settings, get the best model | [`Experiments.ipynb`](Experiments.ipynb): loads the model notebook, trains a list of configurations, ranks them, saves the best one |
| use the best trained model | `models/best_model.pt`, created by `Experiments.ipynb` (see [Using a trained model](#using-a-trained-model)) |

---

## What's in this folder

```
Machine_Translation/
├── Translation_Model_Pytorch.ipynb     ← the model, explained (definitions only)
├── Experiments.ipynb                   ← train and compare configurations, keep the best model
├── Sentence pairs in English-Italian - 2024-07-26.tsv.zip   ← the dataset (~628,000 sentence pairs)
├── experiments/                        ← created by Experiments.ipynb
│   ├── results.csv                     ←   one row per finished experiment
│   └── <experiment name>/model.pt      ←   each experiment's model (not uploaded to GitHub)
├── models/                             ← created by Experiments.ipynb
│   ├── best_model.pt                   ←   the best model, ready to use
│   └── best_model_info.json            ←   which experiment it is and how it scored
├── Data_Intake_and_Preprocessing.ipynb ← earlier, standalone preprocessing pipeline
├── intake_preprocess_utils.ipynb       ← the same preprocessing functions, without the run cells
└── Data/                               ← outputs of the earlier pipeline (not used by the notebooks above)
```

| File | What it's for | Needed? |
|---|---|---|
| `Translation_Model_Pytorch.ipynb` | Defines everything: data preparation, segmentation, RNN / LSTM / Transformer, attention, beam search, training loop, BLEU, and `run_experiment(config)` | **Yes** |
| `Experiments.ipynb` | Runs experiments and keeps the best model | **Yes, to train** |
| `Sentence pairs in English-Italian - 2024-07-26.tsv.zip` | All English–Italian pairs from Tatoeba (July 26, 2024). Read directly from the zip | **Yes** |
| `Data_Intake_and_Preprocessing.ipynb`, `intake_preprocess_utils.ipynb` | The first version of the data pipeline. It makes 5% / 10% / 20% samples and saves PyTorch and TensorFlow data loaders | No |
| `Data/` | Files saved by that first pipeline, with an older token numbering | No |

---

## How to run it

### On your own computer
1. Install the packages. `nltk` is only needed for `stem+bpe`, and spaCy only for `lemma+tags`:
   ```bash
   pip install torch pandas matplotlib nltk spacy
   python -m spacy download it_core_news_sm
   ```
2. Open `Experiments.ipynb` in VS Code or Jupyter, select that Python as the kernel, and run the cells top to bottom.

With the default settings (5% of the data, 10 epochs), one LSTM experiment takes about 50 minutes on a laptop CPU. A Transformer is faster.

### On Google Colab (free GPU, much faster)
1. Put this folder in your Google Drive.
2. Open `Experiments.ipynb` with the **Open in Colab** badge, then choose **Runtime → Change runtime type → GPU**.
3. In the first code cell, set the folder path in `os.chdir(...)`. The same line in the model notebook's setup cell should match.
4. Only for `lemma+tags`: run `!pip install spacy && python -m spacy download it_core_news_sm` first.

If Colab disconnects, run the notebook again. Finished experiments are read from `experiments/results.csv` and skipped.

---

## How an experiment is defined

Every experiment starts from the default settings (`CONFIG` in the model notebook) and lists only what it changes:

```python
EXPERIMENTS = [
  ('lstm_bpe',         {'segmentation': {'english': 'words', 'italian': 'bpe'}}),
  ('bilstm_bpe',       {'segmentation': {'english': 'words', 'italian': 'bpe'}, 'bidirectional': True}),
  ('transformer_bpe',  {'segmentation': {'english': 'words', 'italian': 'bpe'}, 'node_type': 'transformer',
                        'enc_n_layers': 3, 'dec_n_layers': 3, 'learning_rate': 0.0005}),
]
```

Each experiment is trained and scored by **validation BLEU**. The best one is copied to `models/best_model.pt`, and only that model is scored on the **test** sentences. Choosing with validation and testing once at the end keeps the final score honest.

### The most useful settings

| Setting | Default | Options |
|---|---|---|
| `node_type` | `'lstm'` | `'rnn'`, `'lstm'`, `'transformer'` |
| `bidirectional` | `False` | RNN / LSTM only: the encoder also reads each sentence right to left |
| `segmentation` | `{'english': 'words', 'italian': 'bpe'}` | per language: `'words'`, `'bpe'`, `'stem+bpe'`, `'lemma+tags'` |
| `attention` / `sim_method` | `True` / `'scaled_dot'` | RNN / LSTM: `'dot'`, `'scaled_dot'`, `'bahdanau'` |
| `enc_n_layers`, `dec_n_layers` | `1`, `1` | layers in the encoder and decoder |
| `transformer_dim`, `n_heads`, `ff_dim`, `dropout` | `256`, `4`, `512`, `0.1` | Transformer only |
| `sample_frac` | `0.05` | fraction of the ~628k pairs to use |
| `n_epochs`, `learning_rate` | `10`, `0.001` | training length and step size |
| `beam_width` | `3` | candidates kept while translating (`1` = greedy) |

### Segmentation methods
Italian builds many forms from one stem (*parlo, parli, parliamo, parlate…*), so there are four ways of turning words into tokens. The words below are **examples only**, to show how each method behaves:

| Method | *parliamo* ("we speak") becomes | Needs |
|---|---|---|
| `words` | `parliamo` | nothing |
| `bpe` | frequent letter sequences; common words like this one stay whole | nothing |
| `stem+bpe` | `parl@@ iamo` (stemmer cuts off the ending, then BPE) | `nltk` |
| `lemma+tags` | `parlare` + a grammar tag for "1st person plural, present" | `spacy` + `it_core_news_sm` |

`@@` marks a piece that continues into the next one, so `parl@@ iamo` joins back into *parliamo*.

---

## How the model notebook is organized

| Section | What it defines |
|---|---|
| 1. Setup and settings | Imports, GPU check, default `CONFIG` |
| 2. Preparing the Data | Loading, cleaning, train / validation / test split, the four segmentation methods, vocabularies, batches |
| 3. Building blocks | Embedding, RNN cell, LSTM cell |
| 4. Encoder | Reads the English sentence (optionally in both directions) |
| 5. Attention | Dot, scaled dot and Bahdanau scoring |
| 6. Decoder | Writes the Italian sentence one token at a time |
| 7. Seq2Seq | Encoder + decoder; teacher forcing |
| 8. Beam search | Turns the model's scores into a sentence |
| 9. Training loop | Loss, optimizer, gradient clipping, checkpoints |
| 10. Evaluation | Loss and BLEU |
| 11. `run_experiment(config)` | The whole pipeline for one configuration in one call |
| 12. Transformer | Positional encoding, multi-head attention, encoder and decoder layers |

---

## Using a trained model

After running `Experiments.ipynb`, in any notebook:

```python
%run Translation_Model_Pytorch.ipynb
model, eng_vocab, it_vocab = load_checkpoint('models/best_model.pt')
print(translate(model, 'Where are we going?', eng_vocab, it_vocab))
```

---

## Results so far

Early single runs: 1-layer LSTM with attention, 5% of the data, 10 epochs, BLEU on 500 test sentences (0–100, higher is better). The two runs also differed in the minimum word count, so they aren't a controlled comparison. `Experiments.ipynb` is set up to produce one.

| Italian segmentation | BLEU (greedy) | BLEU (beam width 3) |
|---|---|---|
| whole words | 29.4 | 33.1 |
| BPE | 28.6 | 31.6 |

Example translations from the BPE model:

| English | Model output |
|---|---|
| I have to go to sleep. | devo andare a dormire. |
| Where are we going? | dove stiamo andando? |
| What time is it? | a che ora è? |
| We are going to the beach tomorrow. | stiamo andando in spiaggia domani. |

---

## Roadmap
- [x] Batching, beam search, three attention scoring methods, BLEU
- [x] Subword segmentation (BPE, stemmer + BPE, lemma + tags)
- [x] Bidirectional encoder
- [x] Transformer
- [ ] Run the experiment comparison and publish the best model
- [ ] Train on the full dataset with a GPU
- [ ] Pretraining and fine-tuning

## Data source and license
Sentence pairs come from [Tatoeba](https://tatoeba.org), downloaded July 26, 2024. Tatoeba sentences are licensed under [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/).
