from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import itertools

from textblob import TextBlob

import pandas as pd

from .review_base import ReviewBase


class ReviewApp:

    def __init__(self, path_to_base):

        self.base = ReviewBase(path_to_base)

    def build_data_base(self, labeled=None, unlabeled=None):

        self.base.build_and_update(labeled=labeled, unlabeled=unlabeled)

    ### PREPROCESSING
    # text
    #
    #
    @staticmethod
    def _tokenize_lematize(review, lematize=True):
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
        non_stopwords = set(["not", "same", "too", "doesn't", "don't", 'doesn', "didn't", 'didn', "hasn't", 'hasn', "aren't", 'aren', "isn't", 'isn', "shouldn't", 'shouldn', 'wasn', "wasn't", 'weren', "weren't", 'won', "won't"])
        english_stopwords = set([word for word in english_stopwords if word not in non_stopwords])
        tokens = [token for token in tokens if token not in english_stopwords]
        # lematizing the tokens
        if lematize:
            lmtzer = WordNetLemmatizer()
            tokens = [lmtzer.lemmatize(word) for word in tokens]

    @staticmethod
    def _build_vocab(*data_frames, preprocess=False):
        """function that builds a vocabulary from various data_frames
        data_frames is a list of pd.DataFrame that must have a column tokenized_text if preprocess is False and sentence otherwise
        """

        assert all([type(df) is pd.DataFrame for df in data_frames]), "all input data must be dataframes"
        if preprocess:
            assert all(["sentence" in df.columns for df in data_frames]), "all data frames must have a sentence column"
            for i in range(len(data_frames)):
                if not "tokenized_text" in data_frames[i].columns:
                    data_frames[i]["tokenized_text"] = data_frames[i]["sentence"].apply(ReviewApp._tokenize_lematize)

        assert all(["tokenized_text" in df.columns for df in data_frames])
        vocab = list(set(itertools.chain(*[set(itertools.chain(*df.tokenized_text.tolist())) for df in data_frames])))
        vocab_dict = dict((y, x) for x, y in enumerate(vocab))
        return vocab_dict

    # supervised
    #
    #
    @staticmethod
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
        vocab_dict = ReviewApp._build_vocab(*all_df, preprocess=True)

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

    @staticmethod
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
        vocab_dict = ReviewApp._build_vocab(*all_df, preprocess=True)

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

