"""module that takes care of the preprocessing of the sentences for predictions and such"""

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import itertools

from textblob import TextBlob

import pandas as pd


def tokenize_lematize(review, lematize=True):
    if not review:
        return
    # cleaning the review
    replacement_dic = {x: " " for x in ["@", "#", "/", "\n", "\r", "\r", "\b", "\t", "\f", "|"]}
    review = review.translate(str.maketrans(replacement_dic))
    review = " ".join(TextBlob(review.lower()).words)
    # tokenizing
    tkzer = TweetTokenizer(preserve_case=False, reduce_len=True)
    tokens = tkzer.tokenize(review)
    # removing stop words
    english_stopwords = set(stopwords.words("english"))
    non_stopwords = set(["not", "same", "too", "doesn't", "don't", 'doesn', "didn't", 'didn', "hasn't", 'hasn', "aren't", 'aren', "isn't", 'isn', "shouldn't" , 'shouldn', 'wasn',"wasn't", 'weren', "weren't", 'won', "won't"])
    english_stopwords = set([word for word in english_stopwords if word not in non_stopwords])
    tokens = [token for token in tokens if token not in english_stopwords]
    # lematizing the tokens
    if lematize:
        lmtzer = WordNetLemmatizer()
        tokens = [lmtzer.lemmatize(word) for word in tokens]

    return tokens

def build_vocab(*data_frames, preprocess=False):
    """function that builds a vocabulary from various data_frames
    data_frames is a list of pd.DataFrame that must have a column tokenized_text if preprocess is False and sentence otherwise
    """

    assert all([type(df) is pd.DataFrame for df in data_frames]), "all input data must be dataframes"
    if preprocess:
        assert all(["sentence" in df.columns for df in data_frames]), "all data frames must have a sentence column"
        for i in range(len(data_frames)):
            data_frames[i]["tokenized_text"] = data_frames[i]["sentence"].apply(tokenize_lematize)

    assert all(["tokenized_text" in df.columns for df in data_frames])
    vocab = list(set(itertools.chain(*[set(itertools.chain(*df.tokenized_text.tolist())) for df in data_frames])))
    vocab_dict = dict((y, x) for x, y in enumerate(vocab))
    return vocab_dict



