import gensim
import json
import math
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()


def word2int(d, model):
    ints = [model.wv.key_to_index.get(d.lower()) for d in d.split(' ')]
    unk = len(model.wv.key_to_index)
    return [x if x is not None else unk for x in ints]


def main():

    seed = 42

    train_split = 0.75
    val_split = 0.15
    test_split = 0.10

    embed_dim = 300

    print("Reading raw data")
    df = pd.read_csv('../data/imdb/IMDB Dataset.csv')

    data = [d.lower() for d in df['review']]
    data = [gensim.parsing.preprocessing.strip_tags(d) for d in data]
    data = [gensim.parsing.preprocessing.strip_non_alphanum(d) for d in data]
    data = [gensim.parsing.preprocessing.strip_short(d, minsize=2) for d in data]
    data = [d.split(' ') for d in data]

    # create train, test, val splits
    x_train, x_tmp, y_train, y_tmp = train_test_split(df['review'], df['sentiment'], test_size=1-train_split,
                                                    random_state=seed)

    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=test_split/(test_split + val_split),
                                                    random_state=seed)

    train_df = pd.DataFrame(x_train, columns=['review', 'X', 'sentiment', 'split'])
    val_df = pd.DataFrame(x_val, columns=['review', 'X', 'sentiment', 'split'])
    test_df = pd.DataFrame(x_test, columns=['review', 'X', 'sentiment', 'split'])

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    train_df['sentiment'] = y_train
    val_df['sentiment'] = y_val
    test_df['sentiment'] = y_test

    train_data = pd.concat([train_df, val_df])

    print("Creating vocab and word embeddings")
    model = gensim.models.word2vec.Word2Vec(vector_size=embed_dim, min_count=2, epochs=25, workers=4)
    model.build_vocab(train_data['review'].str.split())
    model.train(train_data['review'].str.split(), total_examples=model.corpus_count, epochs=model.epochs)

    # tokenize data
    train_df['X'] = train_df['review'].progress_apply(lambda d: word2int(d, model))
    val_df['X'] = val_df['review'].progress_apply(lambda d: word2int(d, model))
    test_df['X'] = test_df['review'].progress_apply(lambda d: word2int(d, model))

    # add <unk> token
    word_vecs = [model.wv.vectors[index] for index in model.wv.key_to_index.values()]
    rng = np.random.default_rng(seed)
    unk_embed = rng.normal(size=(1, embed_dim), scale=0.1)
    w2v = np.append(word_vecs, unk_embed, axis=0)

    id2word = {v: k for k, v in model.wv.key_to_index.items()}
    id2word[len(model.wv.key_to_index)] = "<unk>"

    print("Saving output files")
    df_out = pd.concat([train_df, val_df, test_df])
    df_out.to_csv("../data/imdb/data_fold0.csv", index=False)

    labels = set(df['sentiment'])
    id2label = {'sentiment': {i: l for i, l in enumerate(labels)}}
    with open('../data/imdb/id2labels_fold0.json', 'w') as f:
        json.dump(id2label, f)

    with open('../data/imdb/id2word.json', 'w') as f:
        json.dump(id2word, f)
    np.save('../data/imdb/word_embeds_fold0.npy', w2v)


if __name__ == "__main__":
    main()
