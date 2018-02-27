""" module that handles all the supervised tasks' preprocessing"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

from .text_preprocessing import build_vocab


def tfidf_preprocessing(train, test=None, additional=None, ngram_range=(1, 3)):
    assert type(train) is pd.DataFrame, "train must be a dataframe"

    # ddoing checks and building vocabulary
    all_df = [train]
    if test is not None:
        assert type(test) is pd.DataFrame, "test must be a dataframe"
        all_df.append(test)
    if additional is not None:
        assert type(additional) is pd.DataFrame, "additional data dataframe must be of type pd.DataFrame"
        all_df.append(additional)
    vocab_dict = build_vocab(*all_df, preprocess=True)

    # building the joined_tokenized columns
    train["joined_tokenized"] = train["tokenized_text"].apply(lambda x: " ".join(x))
    if test is not None:
        test["joined_tokenized"] = test["tokenized_text"].apply(lambda x: " ".join(x))

    # building tfidf matrix
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=ngram_range, vocabulary=vocab_dict)
    x_train = tfidf.fit_transform(train.joined_tokenized.tolist())
    if test is not None:
        x_test = tfidf.fit_transform(test.joined_tokenized.tolist())
        return x_train, x_test
    else:
        return x_train


def bow_preprocessing(train, test=None, additional=None, ngram_range=(1, 3)):

    assert type(train) is pd.DataFrame, "train must be a dataframe"

    # ddoing checks and building vocabulary
    all_df = [train]
    if test is not None:
        assert type(test) is pd.DataFrame, "test must be a dataframe"
        all_df.append(test)
    if additional is not None:
        assert type(additional) is pd.DataFrame, "additional data dataframe must be of type pd.DataFrame"
        all_df.append(additional)
    vocab_dict = build_vocab(*all_df, preprocess=True)

    # building the joined_tokenized columns
    train["joined_tokenized"] = train["tokenized_text"].apply(lambda x: " ".join(x))
    if test is not None:
        test["joined_tokenized"] = test["tokenized_text"].apply(lambda x: " ".join(x))

    # building count matrix
    count = CountVectorizer(max_df=0.95, min_df=2, ngram_range=ngram_range, vocabulary=vocab_dict)
    x_train = count.fit_transform(train.joined_tokenized.tolist())
    if test is not None:
        x_test = count.fit_transform(test.joined_tokenized.tolist())
        return x_train, x_test
    else:
        return x_train
