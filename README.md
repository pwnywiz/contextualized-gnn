# Greek News Classification With Contextualized GNNs

This is the main research method used for the MSc thesis completion. It is based on [TextING](https://github.com/CRIPAC-DIG/TextING) and introduces a pipeline of pre-processing steps and modifications which convert this, from a non-contextual method, to a contextual method with quantized embeddings.

Link to the dataset used for the experiments: [Dataset](https://drive.google.com/file/d/1DT5wZiObwSy2WlFjPWXR_fKJ-iC-6kPZ/view?usp=sharing)

## Requirements

* Python 3.6+
* Tensorflow/Tensorflow-gpu 1.12.0
* Pytorch 1.7.1
* Scipy 1.5.1
* Transformers 3.5.1

## Usage

Extract subword embeddings from the dataset with BERT, cluster them up to 4 centroids and save the mapping in a dictionary as:

    python BERT_embeddings_to_clusters.py

Use euclidian distance to match every subword's embedding to the closest of the precomputed embeddings. Add a suffix to the subword '_N' where N is the index of the matched embedding from the list with:

    python map_words_from_clusters.py

A script that accepts a dictionary with a list of embeddings for a word as in word = [e_1, e_2,...,e_N] and converts it to a flattened dictionary of the format word_N = embed:

    python flatten_word_embeddings.py

Preprocess the text by running `remove_words_bert.py` before building the graphs.

Build graphs from the datasets in `data/corpus/` as:

    python build_graph_bert.py [DATASET] [WINSIZE]

The default sliding window size is 3.

Start training and inference as:

    python train.py [--dataset DATASET] [--learning_rate LR]
                    [--epochs EPOCHS] [--batch_size BATCHSIZE]
                    [--hidden HIDDEN] [--steps STEPS]
                    [--dropout DROPOUT] [--weight_decay WD]

To reproduce the result, large hidden size and batch size are suggested as long as your memory allows.
