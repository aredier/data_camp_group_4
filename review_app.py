import re

import itertools

from datetime import datetime

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from math import sqrt


import pandas as pd

import numpy as np

# to install networkx 2.0 compatible version of python-louvain use:
# pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
from community import community_louvain

import networkx as nx

import spacy

from tqdm import tqdm

from matplotlib import rcParams

from .review_base import ReviewBase


class ReviewApp:
    def __init__(self, path_to_base):

        self._base = ReviewBase(path_to_base)
        self._model = None
        self._model_params = None
        self._vocab = None

    def build_data_base(self, labeled=None, unlabeled=None, log_file=None):

        self._base.build_and_update(labeled=labeled, unlabeled=unlabeled, log_file=log_file)

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
    def _noun_tokenize(sentence, nlp, noun_count=3):

        noun_tokens = [w.lemma_ for w in nlp(sentence.lower()) if w.pos_ == 'NOUN']
        if len(noun_tokens) < noun_count:
            return
        english_stopwords = set(stopwords.words("english"))
        non_stopwords = {"not", "same", "too", "doesn't", "don't", 'doesn', "didn't", 'didn', "hasn't",
                         'hasn', "aren't", 'aren', "isn't", 'isn', "shouldn't", 'shouldn', 'wasn', "wasn't",
                         'weren', "weren't", 'won', "won't"}
        english_stopwords = set([word for word in english_stopwords if word not in non_stopwords])
        noun_tokens = [n for n in noun_tokens if n not in english_stopwords]
        return noun_tokens

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

        # doing checks and building vocabulary
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
        assert model in ["xgb", "logreg", "rf"],

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
                print("performing grid search cross validation")
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

        if model == "rf":
            if do_cv:
                print("performing grid search cross validation")
                p = x_train.shape[1]
                cv_params = {
                    "max_depth": [100, 250, 500],
                    "n_estimators": [10, 100, 1000],
                    "criterion": ["gini", "entropy"],
                    "max_features": ["auto", round(sqrt(p))] # best theoretical subset size for classification
                }
                self._model = RandomForestClassifier()
                gs = RandomizedSearchCV(model, cv_params, n_jobs=-1, scoring="f1", verbose=3)
                gs.fit(x_train, y_train)
                self._model_params = gs.best_params_

            if self._model_params is None:
                self._model_params = {'criterion': "gini",
                                      'max_depth': 250,
                                      'n_estimators': 10,
                                      'max_features': "auto"
                                      }
            
            self._model = RandomForestClassifier()(**self._model_params)
            self._model.fit(x_train, y_train)
        
        if model == "logreg":
            if do_cv:
                print("performing cross validation")
                cv_params = { "C": np.power(10.0, np.arange(-10, 10))}
                self._model = LogisticRegression()
                gs = GridSearchCV(model, cv_params, n_jobs=-1, scoring="f1", verbose=3)
                gs.fit(x_train, y_train)
                self._model_params = gs.best_params_

            if self._model_params is None:
                self._model_params = {'C': 1.0}

            self._model = LogisticRegression()(**self._model_params)
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

    def find_issues(self, start_date=datetime(2018, 1, 1), end_date=None):

        data = self._base.select_detected_issue_from_date(start_date, end_date)
        return data.loc[:, ["date_time", "sentence"]]

    # UNSUPERVISED TASKS
    # graph of words
    #
    #
    #
    @staticmethod
    def _word_neighbors(df, dist=2):
        assert "noun_tokens" in df.columns, "df must be tokenized before distances can be computed"
        return pd.concat([pd.DataFrame([clean_sentence[:-dist], clean_sentence[dist:]]).T for clean_sentence in
                          df.noun_tokens.tolist() if clean_sentence is not None]).rename(columns={0: 'w0', 1: 'w1'}) \
            .reset_index(drop=True)

    def _compute_distances(self, spacy_en_dir="en"):
        nlp = spacy.load(spacy_en_dir)
        df = self._base.get_all_text()
        print("tokenizing into nouns")
        tqdm.pandas()
        df["noun_tokens"] = df.sentence.progress_apply(lambda text: ReviewApp._noun_tokenize(text, nlp))
        print("building distances")
        distances = ReviewApp._word_neighbors(df, 1).assign(weight=2).append(
            ReviewApp._word_neighbors(df, 1).assign(weight=1))
        distances = distances.groupby(['w0', 'w1']).weight.sum().reset_index()
        return distances

    def _build_gof(self, spacy_en_dir="en"):
        data_graph_of_words = self._compute_distances(spacy_en_dir)
        print("building graph")
        graph_of_words = nx.from_pandas_edgelist(data_graph_of_words, source='w0', target='w1', edge_attr='weight',
                                                 create_using=nx.Graph())
        return graph_of_words

    # graph clustering
    # The code is taken from the link below
    # https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
    @staticmethod
    def _community_layout(g, partition):
        """
        Compute the layout for a modular graph.


        Arguments:
        ----------
        g -- networkx.Graph or networkx.DiGraph instance
            graph to plot

        partition -- dict mapping int node -> int community
            graph partitions


        Returns:
        --------
        pos -- dict mapping int node -> (float x, float y)
            node positions

        """

        pos_communities = ReviewApp._position_communities(g, partition, scale=3.)

        pos_nodes = ReviewApp._position_nodes(g, partition, scale=1.)

        # combine positions
        pos = dict()
        for node in g.nodes():
            pos[node] = pos_communities[node] + pos_nodes[node]

        return pos

    @staticmethod
    def _position_communities(g, partition, **kwargs):

        # create a weighted graph, in which each node corresponds to a community,
        # and each edge weight to the number of edges between communities
        between_community_edges = ReviewApp._find_between_community_edges(g, partition)

        communities = set(partition.values())
        hypergraph = nx.DiGraph()
        hypergraph.add_nodes_from(communities)
        for (ci, cj), edges in between_community_edges.items():
            hypergraph.add_edge(ci, cj, weight=len(edges))

        # find layout for communities
        pos_communities = nx.spring_layout(hypergraph, **kwargs)

        # set node positions to position of community
        pos = dict()
        for node, community in partition.items():
            pos[node] = pos_communities[community]

        return pos

    @staticmethod
    def _find_between_community_edges(g, partition):

        edges = dict()

        for (ni, nj) in g.edges():
            ci = partition[ni]
            cj = partition[nj]

            if ci != cj:
                try:
                    edges[(ci, cj)] += [(ni, nj)]
                except KeyError:
                    edges[(ci, cj)] = [(ni, nj)]

        return edges

    def _position_nodes(g, partition, **kwargs):
        """
        Positions nodes within communities.
        """

        communities = dict()
        for node, community in partition.items():
            try:
                communities[community] += [node]
            except KeyError:
                communities[community] = [node]

        pos = dict()
        for ci, nodes in communities.items():
            subgraph = g.subgraph(nodes)
            pos_subgraph = nx.spring_layout(subgraph, **kwargs)
            pos.update(pos_subgraph)

        return pos

    @staticmethod
    def _get_partition_resume(graph, partition, n_words=4):

        partition_df = pd.DataFrame.from_dict(partition, orient="index").rename(columns={0: 'group'})
        n_words = min(n_words, min(partition_df["group"].value_counts().values))
        groups_df = pd.DataFrame(columns=set(partition_df["group"]), index=list(range(1, n_words + 1)))
        for group in set(partition_df["group"]):
            subgraph = graph.subgraph(partition_df[partition_df["group"] == group].index.values)
            groups_df[group] = pd.DataFrame.from_dict([nx.pagerank(G=subgraph, alpha=0.99)]).T.rename(
                columns={0: 'pagerank'}) \
                                   .sort_values("pagerank", ascending=False).index.values[:n_words]
        return groups_df

    def do_gof_anlysis(self, key_word="issue", draw=True, space_en_dir="en"):

        graph_of_words = self._build_gof(spacy_en_dir=space_en_dir)
        G = nx.ego_graph(G=graph_of_words, radius=1, n=key_word)
        print("building partition")
        partition = community_louvain.best_partition(G)
        if draw:
            pos = ReviewApp._community_layout(g=G, partition=partition)
            rcParams['figure.figsize'] = (40, 40)
            nx.draw(G, pos, node_color=list([partition[key] for key in list(G.nodes)]),
                    labels=dict((n, n) for n, d in G.nodes(data=True)), font_color='black', font_size=8, font_weight='bold',
                    edge_color='lightgray')

        print(self._get_partition_resume(graph_of_words, partition, 6))

