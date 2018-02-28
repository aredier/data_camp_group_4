from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import itertools

from textblob import TextBlob

import pandas as pd

from .review_base import ReviewBase

from tqdm import tqdm



class ReviewApp:

    def __init__(self, path_to_base):

        self.base = ReviewBase(path_to_base)
        self._model = None
        self._model_params = None

    def build_data_base(self, labeled=None, unlabeled=None):

        self.base.build_and_update(labeled=labeled, unlabeled=unlabeled)

    # PREPROCESSING
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
        return tokens

    @staticmethod
    def _build_vocab(*data_frames, preprocess=False):
        """function that builds a vocabulary from various data_frames
        data_frames is a list of pd.DataFrame that must have a column tokenized_text if preprocess is False and sentence otherwise
        """

        assert all([type(df) is pd.DataFrame for df in data_frames]), "all input data must be dataframes"
        if preprocess:
            tqdm.pandas()
            assert all(["sentence" in df.columns for df in data_frames]), "all data frames must have a sentence column"
            for i in range(len(data_frames)):
                if not "tokenized_text" in data_frames[i].columns:
                    print("- tokenizing data_frame number {}".format(i))
                    data_frames[i]["tokenized_text"] = data_frames[i]["sentence"].progress_apply(ReviewApp._tokenize_lematize)

        assert all(["tokenized_text" in df.columns for df in data_frames])
        print("- building vocab")
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
        data = self.base.get_train().drop(["sentence_id", "predicted", "id"], axis=1)
        x = self.bow_preprocessing(data, additional=self.base.get_all_text())
        y = data["issue"]
        if do_test_analysis:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_prop)
        else:
            x_train = x
            y_train = y

        print("training model")
        if model == "xgb":
            if do_cv:
                pass

            if self._model_params is None:
                self._model_params = {'gamma': 0.0001, 'max_depth': 250, 'n_estimators': 500}

            self._model = XGBClassifier(self._model_params)
            self._model.fit(x_train, y_train)

        if do_test_analysis:
            print("Performing test anlysis")
            y_pred = self._model.predict(x_test)
            print(classification_report(y_test, y_pred))
