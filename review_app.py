import re

import itertools

from datetime import datetime

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report


from xgboost import XGBClassifier

import pandas as pd

import numpy as np

from tqdm import tqdm


from .review_base import ReviewBase


class ReviewApp:

    def __init__(self, path_to_base):

        self._base = ReviewBase(path_to_base)
        self._model = None
        self._model_params = None
        self._vocab = None

    def build_data_base(self, labeled=None, unlabeled=None):

        self._base.build_and_update(labeled=labeled, unlabeled=unlabeled)
    def update_data_base(self):

        self._base.update()

    # PREPROCESSING
    # text
    #
    #
    @staticmethod
    def _tokenize_lematize(review, lematize=True):
        if not review:
            return
        # cleaning the review
        review = str(review)
        review = review.lower()
        review = review.strip(' \t\n\r')
        review = re.sub("[@#$!%^&*()_+|~=`{}\\:;<>?,.\/]", " ", review)

        # tokenizing
        tkzer = TweetTokenizer(preserve_case=False, reduce_len=True)
        tokens = tkzer.tokenize(review)
        # removing stop words
        english_stopwords = set(stopwords.words("english"))
        non_stopwords = {"not", "same", "too", "doesn't", "don't", 'doesn', "didn't", 'didn', "hasn't",
                             'hasn', "aren't", 'aren', "isn't", 'isn', "shouldn't", 'shouldn', 'wasn', "wasn't",
                             'weren', "weren't", 'won', "won't"}
        english_stopwords = set([word for word in english_stopwords if word not in non_stopwords])
        tokens = [token for token in tokens if token not in english_stopwords]
        # lematizing the tokens
        if lematize:
            lmtzer = WordNetLemmatizer()
            tokens = [lmtzer.lemmatize(word) for word in tokens]
        return tokens

    @staticmethod
    def _tokenize_df(df, target="sentence"):
        tqdm.pandas()
        assert type(target) is str, "target must be a string"
        assert target in df.columns, "dataframe must have a {} column (user specified) to tokenize".format(target)
        df["tokenized_text"] = df[target].progress_apply(ReviewApp._tokenize_lematize)
        return df

    def _build_vocab(self, *data_frames, preprocess=False):
        """function that builds a vocabulary from various data_frames
        data_frames is a list of pd.DataFrame that must have a column tokenized_text if preprocess is False and
        sentence otherwise
        """
        if self._vocab is not None:
            return self._vocab
        data_frames = list(data_frames)
        assert all([type(df) is pd.DataFrame for df in data_frames]), "all input data must be dataframes"
        if preprocess:
            tqdm.pandas()
            assert all(["sentence" in df.columns for df in data_frames]), "all data frames must have a sentence column"
            for i in range(len(data_frames)):
                if "tokenized_text" not in data_frames[i].columns:
                    print("- tokenizing data_frame number {}".format(i))
                    data_frames[i] = self._tokenize_df(data_frames[i])

        assert all(["tokenized_text" in df.columns for df in data_frames])
        print("- building vocab")
        vocab = list(set(itertools.chain(*[set(itertools.chain(*df.tokenized_text.tolist())) for df in data_frames])))
        vocab_dict = dict((y, x) for x, y in enumerate(vocab))
        self._vocab = vocab_dict
        return self._vocab

    # supervised
    #
    #
    def tfidf_preprocessing(self, train, test=None, additional=None, ngram_range=(1, 3)):

        assert type(train) is pd.DataFrame, "train must be a dataframe"

        # ddoing checks and building vocabulary
        all_df = [train]
        if test is not None:
            assert type(test) is pd.DataFrame, "test must be a dataframe"
            all_df.append(test)
        if additional is not None:
            assert type(additional) is pd.DataFrame, "additional data dataframe must be of type pd.DataFrame"
            all_df.append(additional)
        vocab_dict = self._build_vocab(*all_df, preprocess=True)

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

    def bow_preprocessing(self, train, test=None, additional=None, ngram_range=(1, 3)):

        assert type(train) is pd.DataFrame, "train must be a dataframe"

        # ddoing checks and building vocabulary
        all_df = [train]
        if test is not None:
            assert type(test) is pd.DataFrame, "test must be a dataframe"
            all_df.append(test)
        if additional is not None:
            assert type(additional) is pd.DataFrame, "additional data dataframe must be of type pd.DataFrame"
            all_df.append(additional)
        vocab_dict = self._build_vocab(*all_df, preprocess=True)

        # checking the data is preprocessed
        if "tokenized_text" not in train.columns:
            print("- tokenizing train data frame")
            train = self._tokenize_df(train)
        if test is not None and "tokenized_text" not in test.columns:
            print("- tokenizing test data frame")
            test = self._tokenize_df(test)

        # building the joined_tokenized columns
        print("- rebuilding clean data ")
        tqdm.pandas()
        train["joined_tokenized"] = train["tokenized_text"].progress_apply(lambda x: " ".join(x))
        if test is not None:
            test["joined_tokenized"] = test["tokenized_text"].progress_apply(lambda x: " ".join(x))

        # building count matrix
        print("- building bow")
        count = CountVectorizer(max_df=0.95, min_df=2, ngram_range=ngram_range, vocabulary=vocab_dict)
        x_train = count.fit_transform(train.joined_tokenized.tolist())
        if test is not None:
            x_test = count.fit_transform(test.joined_tokenized.tolist())
            return x_train, x_test
        else:
            return x_train

    # SUPERVISED TASKS
    #
    #
    def train_model(self, model="xgb", do_test_analysis=True, test_prop=0.33, do_cv=False):
        """trains the model"""
        assert model in ["xgb"], "Only XGBoost suported so far"

        print("doing preprocessing (tokenizing, lematizing, bow, ...)")
        print("- getting data")
        data = self._base.get_train().drop(["sentence_id", "predicted", "id"], axis=1)
        x = self.bow_preprocessing(data, additional=self._base.get_all_text())
        y = data["issue"]
        if do_test_analysis:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_prop)
        else:
            x_train = x
            y_train = y

        print("training model")
        if model == "xgb":
            if do_cv:
                print("performing greid search cross validation")
                cv_params = {
                    "max_depth": [100, 250, 500],
                    "n_estimators": [100, 500, 1000],
                    "gamma": [0.01, 0.001, 0.0001]
                }
                self._model = XGBClassifier()
                gs = RandomizedSearchCV(model, cv_params, n_jobs=-1, scoring="f1", verbose=3)
                gs.fit(x_train, y_train)
                self._model_params = gs.best_params_

            if self._model_params is None:
                self._model_params = {'gamma': 0.0001, 'max_depth': 250, 'n_estimators': 500}

            self._model = XGBClassifier(**self._model_params)
            self._model.fit(x_train, y_train)

        if do_test_analysis:
            print("Performing test anlysis")
            y_pred = self._model.predict(x_test)
            print(classification_report(y_test, y_pred))

    def update_predictions(self):
        """updates the predictions in the data base"""
        assert self._model is not None, "model must be fitted or loaded before predictions are possible"
        data = self._base.get_not_predicted()
        x = self.bow_preprocessing(data)
        print("- performing predictions")
        y = self._model.predict(x)
        result_df = pd.DataFrame(np.array([data["id"], y]).T, columns=["sentence_id", "issue"])
        print("updating data base")
        self._base.update_predictions(result_df)

    def find_issues(self, start_date=datetime(2018,1,1), end_date=None):

        data = self._base.select_detected_issue_from_date(start_date, end_date)
        return data.loc[:,["date_time", "sentence"]]


