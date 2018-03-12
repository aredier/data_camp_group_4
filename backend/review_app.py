import re

import itertools

from datetime import datetime

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, classification_report

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
    """backend app for phones defects alerts based on user reviews thorough out the web"""

    def __init__(self, path_to_base):
        """
        initialisation of the app


         Arguments:
        ----------
        path_to_base -- str : location of existing database or desired one
        """

        self._base = ReviewBase(path_to_base)
        self._models = dict()
        self._model_params = dict()
        self._vocab = None
        try:
            self.predicted = self._base.get_not_predicted().shape[0] == 0
        except pd.io.sql.DatabaseError:
            self.predicted = False

    def issue_type_count(self, options):
        return self._base.get_issue_type_count(options)

    @property
    def issue_categories(self):
        return self._base.issue_categories

    def build_data_base(self, labeled=None, unlabeled=None, log_file=None):
        """
        building app"s databse from provided csv's as well as new data scraped from the web


         Arguments:
        ----------
        labeled -- str : path to csv must be formated into sentences with a column text and a column per issue type
        unlabeled -- str : path to csv, must be formated into sentences
        log_file -- str : path to scraper log file
        """

        self._base.build_and_update(labeled=labeled, unlabeled=unlabeled, log_file=log_file)

    def update_data_base(self, log_file):
        """
        update the app data base by running scrapers


         Arguments:
        ----------
        log_file -- str path to scraper log_file
        """

        self._base.update(log_file=log_file)

    # PREPROCESSING
    # text
    #
    #
    @staticmethod
    def _tokenize_lematize(review, lematize=True):
        """
        tokenize and lematize a sentence into words with stop words removal


        Arguments:
        ----------
        review -- str : sentence to be tokenized
        lematize -- bool : wether to lematize or not

        Returns:
        ----------
        tokens -- list : tokens
        """

        assert type(review) is str, "can only tokenize strings"
        assert type(lematize) is bool, "lematize must be boolean"
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
    def _graph_tokenize(sentence, nlp, noun_count=5):
        """
        tokenizing sentences into words for graph of words (keeping only nouns and adjectives).


        Arguments:
        ----------
        sentence -- str : sentence to be tokenized
        nlp -- spacy : spacy en environment
        noun_count -- int : min number of entities per sentence

        Returns:
       ----------
        noun_tokens -- list : tokens
        """

        noun_tokens = [w.lemma_ for w in nlp(sentence.lower()) if (w.pos_ == 'NOUN' or w.pos_ == "ADJ")]
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
        """
        tokenizing the columb target of a data frame with progress bar


        Arguments:
        ----------
        df -- pd.DataFrame : data frame in which to pick the column to be tokenized
        target -- str : target column in the data frame

        Returns:
        ----------
        df -- pd.DataFrame : mutated data frame with the column 'tokenized_text' added (or replaced)
        """
        tqdm.pandas()
        assert type(target) is str, "target must be a string"
        assert target in df.columns, "dataframe must have a {} column (user specified) to tokenize".format(target)
        df["tokenized_text"] = df[target].progress_apply(ReviewApp._tokenize_lematize)
        return df

    def _build_vocab(self, *additionals, preprocess=False):
        """
        function that builds a vocabulary from various data_frames


        Arguments:
        ----------
        data_frames -- list : list of pd.DataFrame that must have a column tokenized_text
        if preprocess is False and sentence otherwise
        preprocesss -- bool : wether to preprocess the data frames or not

        Returns:
        ----------
        _vocab -- dict : the build vocabulary of the app (which is also set as app atribute)
        """
        if self._vocab is not None:
            return self._vocab
        data_frames = [self._base.get_all_text(), *additionals]
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

    # supervised preprocessing
    #
    #
    def tfidf_preprocessing(self, train, test=None, additional=None, ngram_range=(1, 3)):
        """
        performs tfidf preprocessing on the desired train and test dataframes


         Arguments:
        ----------
        train -- pd.DataFrame : train data frame
        test -- pd.DataFrame : test data frame
        additional -- pd.DataFRame : additional data to use when building app vocab
        ngram_range -- tuple : n_grames to be used for the tfidf vecorization

        Returns:
        ----------
        x_train -- sparse matrix tfidf matrix of the train set
        x_test -- sparse matrix if test is specified (tfidf matrix of the test set)
        """

        assert type(train) is pd.DataFrame, "train must be a dataframe"

        # doing checks and building vocabulary
        if test is not None:
            assert type(test) is pd.DataFrame, "test must be a dataframe"
        if additional is not None:
            assert type(additional) is pd.DataFrame, "additional data dataframe must be of type pd.DataFrame"
        vocab_dict = self._build_vocab(preprocess=True)

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
        """
        performs bag of words (bow) preprocessing on the desired train and test dataframes


        Arguments:
        ----------
        train -- pd.DataFrame : train data frame
        test -- pd.DataFrame : test data frame
        additional -- pd.DataFRame : additional data to use when building app vocab
        ngram_range -- tuple : n_grames to be used for the tfidf vecorization

        Returns:
        ----------
        x_train -- sparse matrix bow matrix of the train set
        x_test -- sparse matrix if test is specified (bow matrix of the test set)
        """

        assert type(train) is pd.DataFrame, "train must be a dataframe"

        # ddoing checks and building vocabulary
        all_df = [train]
        if test is not None:
            assert type(test) is pd.DataFrame, "test must be a dataframe"
            all_df.append(test)
        if additional is not None:
            assert type(additional) is pd.DataFrame, "additional data dataframe must be of type pd.DataFrame"
            all_df.append(additional)
        vocab_dict = self._build_vocab(preprocess=True)

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
    def _inner_train(self, x, y, model="xgb", do_cv=False):
        """
         does train on all the issues of the data for a specified model. Can perform cross validation for all the models


         Arguments:
        ----------
        x -- pd.DataFrame or np.array : train input data
        y -- pd.DataFrame : train output data
        model -- str : model to be used ('xgb', 'rf', 'logreg')
        do_cv -- bool : wether to perform cross validation
         """

        if model == "xgb":
            for col in y.columns:
                print("- training on {}".format(col))
                if do_cv:
                    print("performing grid search cross validation")
                    cv_params = {
                        "max_depth": [100, 250, 500],
                        "n_estimators": [100, 500, 1000],
                        "gamma": [0.01, 0.001, 0.0001]
                    }
                    self._models[col] = XGBClassifier()
                    gs = RandomizedSearchCV(self._models[col], cv_params, n_jobs=-1, scoring="f1", verbose=3)
                    gs.fit(x, y[col])
                    self._model_params[col] = gs.best_params_
                    self._models[col] = gs.best_estimator_

                else:
                    if col not in self._model_params.keys():
                        self._model_params[col] = {'gamma': 0.0001, 'max_depth': 250, 'n_estimators': 500}
                    self._models[col] = XGBClassifier(**self._model_params[col])
                    self._models[col].fit(x, y[col])

        elif model == "rf":
            for col in y.columns:
                print("- training on {}".format(col))
                if do_cv:
                    print("performing grid search cross validation")
                    p = x.shape[1]
                    cv_params = {
                        "max_depth": [100, 250, 500],
                        "n_estimators": [10, 100, 1000],
                        "criterion": ["gini", "entropy"],
                        "max_features": ["auto", round(sqrt(p))]  # best theoretical subset size for classification
                    }
                    self._models[col] = RandomForestClassifier()
                    gs = RandomizedSearchCV(self._models[col], cv_params, n_jobs=-1, scoring="f1", verbose=3)
                    gs.fit(x, y[col])
                    self._model_params[col] = gs.best_params_
                    self._models[col] = gs.best_estimator_
                else:
                    if col not in self._model_params.keys():
                        self._model_params[col] = {'criterion': "gini",
                                                   'max_depth': 250,
                                                   'n_estimators': 10,
                                                   'max_features': "auto"
                                                   }

                    self._models[col] = RandomForestClassifier(**self._model_params[col])
                    self._models[col].fit(x, y[col])

        elif model == "logreg":
            for col in y.columns:
                print("- training on {}".format(col))
                if do_cv:
                    print("performing cross validation")
                    cv_params = {"C": np.power(10.0, np.arange(-10, 10))}
                    self._models[col] = LogisticRegression()
                    gs = GridSearchCV(self._models[col], cv_params, n_jobs=-1, scoring="f1", verbose=3)
                    gs.fit(x, y[col])
                    self._model_params[col] = gs.best_params_
                    self._models[col] = gs.best_estimator_
                else:
                    if col not in self._model_params.keys():
                        self._model_params[col] = {'C': 1.0}

                    self._models[col] = LogisticRegression(**self._model_params[col])
                    self._models[col].fit(x, y[col])

    def train_model(self, model="xgb", do_test_analysis=True, format_analysis=False, test_prop=0.33, do_cv=False):
        """
        trains app's inner model


        Arguments:
        ----------
        model -- str : model to be used (xgb, rf or logreg)
        do_test_analysis -- bool : wether to do test analysis or not
        test_prop -- float : test proportion to be used if do_test_analysis is true
        do_cv -- bool : wether to perform cross_validation or not
        """

        assert model in ["xgb", "logreg", "rf"], "only XGBoost, logistic regression and random forest suported, {} was provided".format(model)
        assert self._models == dict(), "only use this method for first train, please use retrain for retraining models"

        print("doing preprocessing (tokenizing, lematizing, bow, ...)")
        print("- getting data")
        data = self._base.get_train().drop(["sentence_id", "date_time", "predicted", "id"], axis=1)
        x = self.bow_preprocessing(data, additional=self._base.get_all_text())
        y = data.drop(["sentence", "tokenized_text", "joined_tokenized"], axis=1)
        if do_test_analysis:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_prop)
        else:
            x_train = x
            y_train = y

        print("training model")
        self._inner_train(x_train, y_train, model, do_cv=do_cv)

        if do_test_analysis:
            print("Performing test anlysis")
            for col in y_test.columns:
                print("for {}".format(col))
                y_pred = self._models[col].predict(x_test)
                if format_analysis:
                    yield(classification_report(y_test[col], y_pred))
                else:
                    yield col, precision_recall_fscore_support(y_test[col], y_pred)

    def retrain(self, keep_params=False, *args, **kargs):
        """
        retrains the inner model of the app


        Arguments:
        ----------
        keep_params -- bool : wether to keep the parameters of the inner model
        args -- list : train_model positional arguments
        kargs -- dict : train_model optional arguments
        """

        self.predicted = False
        if not keep_params:
            self._model_params = dict()
        self._models = dict()
        for value in self.train_model(*args, **kargs):
            yield value

    def _predict(self, x):
        """
         performing predictions


         Arguments:
        ----------
        x -- pd.Dataframe or np.array : bag of word of entry
         """
        res = pd.DataFrame()
        for col in tqdm(self._models.keys()):
            res = pd.concat([res, pd.DataFrame(self._models[col].predict(x), columns=[col])], axis=1)
        return res

    def update_predictions(self):
        """
        updates the predictions in the data base
        """


        assert self._models != dict(), "model must be fitted or loaded before predictions are possible"
        self._base.delete_predictions()
        data = self._base.get_not_predicted()
        i = 0
        while data.shape[0] != 0:
            print("UPDATING PREDICTIONS FOR CHUNK {}".format(i))
            x = self.bow_preprocessing(data)
            print("- performing predictions")
            y =  self._predict(x.todense())
            y_val = y.values
            ids = data["id"].values.reshape(-1,1)
            if y_val.shape[0] != ids.shape[0]:
                raise RuntimeError("internal error on binding results to sentence ids")
            result_df = pd.DataFrame(np.concatenate((ids, y_val), axis=1), columns=["sentence_id", *y.columns])
            print("- updating data base")
            self._base.update_predictions(result_df)

            i += 1
            data = self._base.get_not_predicted()

        self.predicted = True

    def find_issues(self, start_date=datetime(2018, 1, 1), end_date=None):
        """
        find new issues predicted by the app


        Arguments:
        ----------
        start_date -- datetime.date or datetime.datetime : date from which to find issues
        end_date -- datetime.date or datetime.datetime : date at which to en search
        """
        data = self._base.select_detected_issue_from_date(start_date, end_date)
        return data

    # UNSUPERVISED TASKS
    # graph of words
    #
    #
    #
    @staticmethod
    def _word_neighbors(df, dist=2):
        """
        computes the neighbours for a data frame of tokens


        Arguments:
        ----------
        df -- pd.DataFrame : data frame from which to build neighbours, must contain a column 'noun_tokens'
        dist -- int : distance at which to word are considered to be neighbors

        Returns:
        ----------
        new_df -- pd.DataFrame : data frame with to column ('w0' and 'w1') where to words in the same line are
        neighbors
        """
        assert "noun_tokens" in df.columns, "df must be tokenized before distances can be computed"
        return pd.concat([pd.DataFrame([clean_sentence[:-dist], clean_sentence[dist:]]).T for clean_sentence in
                          df.noun_tokens.tolist() if clean_sentence is not None]).rename(columns={0: 'w0', 1: 'w1'}) \
            .reset_index(drop=True)

    def _compute_distances(self, spacy_en_dir="en"):
        """
        builds the distances between words for graph construction


        Arguments:
        ----------
        spacy_en_dir -- str : directory of the spacy english environment

        Returns:
        ----------
        distances -- pd.DataFrame : data frame with the edge distance between each words
        """
        nlp = spacy.load(spacy_en_dir)
        df = self._base.get_all_text()
        print("tokenizing")
        tqdm.pandas()
        df["noun_tokens"] = df.sentence.progress_apply(lambda text: ReviewApp._graph_tokenize(text, nlp))
        print("building distances")
        distances = ReviewApp._word_neighbors(df, 1).assign(weight=2).append(
            ReviewApp._word_neighbors(df, 1).assign(weight=1))
        distances = distances.groupby(['w0', 'w1']).weight.sum().reset_index()
        return distances

    def _build_gof(self, spacy_en_dir="en"):
        """
        builds graph of words for the all the sentences in the data frame


        Arguments:
        ----------
        spacy_en_dir -- str : path to spacy english model module

        Returns:
        ----------
        graph_of_words -- nx.Graph : graph of words of all the sentences within the database
        """
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

        pos_communities = ReviewApp._position_communities(g, partition, scale=5.)

        pos_nodes = ReviewApp._position_nodes(g, partition, scale=1.)

        # combine positions
        pos = dict()
        for node in g.nodes():
            pos[node] = pos_communities[node] + pos_nodes[node]

        return pos

    @staticmethod
    def _position_communities(g, partition, **kwargs):
        """
        position the comunities in for clusterd graph


        Arguments:
        ----------
        g -- nx.Graph : graph that is to be printed
        partition -- dic : graph clustering to be applied
        kwargs -- dict : additional layout named arguments

        Returns:
        ----------
        pos -- dict : communities positions
        """

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
        """
        finds edges between partions (communities) for clusterd graph


         Arguments:
        ----------
        g -- nx.Graph : graph that is to be printed
        partition -- dic : graph clustering to be applied

        Arguments:
        ----------
        edges -- dict : between comunity edges
        """

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

    def _position_nodes(self, partition, **kwargs):
        """
        Positions nodes within communities.

        Arguments:
        ----------
        g -- nx.Graph : graph that is to be printed
        partition -- dic : graph clustering to be applied

        Returns:
        ----------
        pos -- dict : within communities positions
        """

        communities = dict()
        for node, community in partition.items():
            try:
                communities[community] += [node]
            except KeyError:
                communities[community] = [node]

        pos = dict()
        for ci, nodes in communities.items():
            subgraph = self.subgraph(nodes)
            pos_subgraph = nx.spring_layout(subgraph, **kwargs)
            pos.update(pos_subgraph)

        return pos

    @staticmethod
    def _get_partition_resume(graph, partition, n_words=4):
        """
        gets the resume (most cental words) of a graph of word partition

        Arguments:
        ----------
        graph -- nx.Graph : graph on which partition is applied
        partition -- dict : partion to be analysed
        n_words -- int : number of words to be kept in resume

        Returns:
        ----------
        groups_df -- pd.DataFrame : resume data frame
        """

        partition_df = pd.DataFrame.from_dict(partition, orient="index").rename(columns={0: 'group'})
        n_words = min(n_words, min(partition_df["group"].value_counts().values))
        groups_df = pd.DataFrame(columns=set(partition_df["group"]), index=list(range(1, n_words + 1)))
        for group in set(partition_df["group"]):
            subgraph = graph.subgraph(partition_df[partition_df["group"] == group].index.values)
            groups_df[group] = pd.DataFrame.from_dict([nx.pagerank(G=subgraph, alpha=0.99)]).T.rename(
                columns={0: 'pagerank'}) \
                                   .sort_values("pagerank", ascending=False).index.values[:n_words]
        return groups_df

    def do_gof_anlysis(self, key_word="issue", draw=True, out=False, space_en_dir="en"):
        """
        performs the graph of words analysis


        Arguments:
        ----------
        key_word -- str : keyword to use to extract subgraph on which clustering is performed
        draw -- bool : wether to draw subgraph with clusters or not
        out -- bool : wether to output results or not
        spacy_en_dir : sapcy english environement directory "en" should be sufficient if symlink is set
                       otherwise specify package directory

        Returns:
        ----------
        G -- networkix.Graph : subgraph on which the clustering is done
        partition -- dict : clustering of the graph
        """

        graph_of_words = self._build_gof(spacy_en_dir=space_en_dir)
        G = nx.ego_graph(G=graph_of_words, radius=1, n=key_word)
        print("building partition")
        partition = community_louvain.best_partition(G)
        if draw:
            pos = ReviewApp._community_layout(g=G, partition=partition)
            rcParams['figure.figsize'] = (40, 40)
            nx.draw(G, pos, node_color=list([partition[key] for key in list(G.nodes)]),
                    labels=dict((n, n) for n, d in G.nodes(data=True)), font_color='black', font_size=6,
                    font_weight='bold',
                    edge_color='lightgray', node_size=35)

        print(self._get_partition_resume(graph_of_words, partition, 6))

        if out:
            return G, partition
