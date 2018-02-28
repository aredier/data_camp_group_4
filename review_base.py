import sqlite3
from datetime import date, datetime

import arrow
import pandas as pd
from scrapy.crawler import CrawlerProcess
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from .scrapy_scrapers.scrapy_scrapers.spiders.best_buy_spider import BestBuySpider
from .models import Base, Sentences, Reviews


class ReviewBase:
    """abstraction of the review database"""


    def __init__(self, path_to_base):

        assert type(path_to_base) is str, "path must be string"
        self._path = path_to_base

        self._engine = create_engine("sqlite:///" + self._path, echo=False)
        self._session_maker = sessionmaker(bind=self._engine)

        self._conn = sqlite3.connect(self._path)

    def _add_issue(self, label):
        """adds an issue colum to the issues table """

        try:
            self._conn.execute("ALTER TABLE issues ADD COLUMN {} INTEGER".format(label))
        except sqlite3.OperationalError:
            pass

    ### BUILDING AND UPDATING THE DATABASE
    #
    #
    @staticmethod
    def _insert_unlabeled(session, row, insert_date):
        """inserts an issue from the unlabled dataframe"""


        review = Reviews(date_time=insert_date)
        session.add(review)
        session.commit()
        sentence = Sentences(sentence=row["text"], review_id=review.id, review_pos=0)
        session.add(sentence)
        session.commit()

    @staticmethod
    def _insert_labeled( session, row, IssueClass, insert_date):
        """ inserts an issue from the labeled datafrmae in the database"""

        review = Reviews(date_time=insert_date)
        session.add(review)
        session.commit()
        sentence = Sentences(sentence=row["text"], review_id=review.id, review_pos=0)
        session.add(sentence)
        session.commit()
        issue = IssueClass(sentence_id=sentence.id, predicted=False)
        row = row.iloc[1:]
        for issue_name in row.index:
            setattr(issue, issue_name, row[issue_name])
        session.add(issue)
        session.commit()

    def _build(self, labeled=None, unlabeled=None):
        """builds the data base and populates it with reviews form labeled and unlabeled data frame if specified"""
        tqdm.pandas()

        Base.metadata.create_all(self._engine)

        insert_date = date(2018, 1, 1)

        if labeled:
            session = self._session_maker()
            print("importing labeled data")
            labeled = pd.read_csv(labeled)
            for label in labeled:
                if label == "text":
                    continue
                self._add_issue(label)
            Base2 = automap_base()
            Base2.prepare(self._engine, reflect=True)
            Issue = Base2.classes.issues
            labeled.progress_apply(lambda row: self._insert_labeled(session, row, Issue, insert_date), axis=1)
            session.close()

        if unlabeled:
            print("importing unalabeled data")
            session = self._session_maker()
            unlabeled = pd.read_csv(unlabeled)
            unlabeled.progress_apply(lambda row: self._insert_unlabeled(session, row, insert_date), axis=1)

    def update(self, update_date=None):

        if not update_date:
            session = self._session_maker()
            update_date = session.query(Reviews).distinct(Reviews.date_time).order_by(Reviews.date_time.desc()).first()
            if not update_date:
                update_date = datetime(2017, 1, 1).replace(tzinfo=None)
            else:
                update_date = update_date.date_time.replace(tzinfo=None)

        process = CrawlerProcess({
            'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:57.0) Gecko/20100101 Firefox/57.0',
            "path": self._path,
            "date_time": update_date,
            "LOG_LEVEL": "INFO",
            "ITEM_PIPELINES": {
                'group_4_backend.scrapy_scrapers.scrapy_scrapers.pipelines.ScrapyScrapersPipeline': 300,
            },
            "ROBOTSTXT_OBEY": False,
            "CONCURRENT_REQUESTS": 32,
            "CONCURRENT_REQUESTS_PER_DOMAIN ": 32,
            "AUTOTHROTTLE_ENABLED": False,
        })

        process.crawl(BestBuySpider)
        process.start()

    def build_and_update(self, labeled=None, unlabeled=None, update=True):

        print("\n BUILDING BASE FROM CSVs")
        self._build(labeled=labeled, unlabeled=unlabeled)
        if update:
            print("\nSCRAPPING NEW DATA")
            self.update()

    # SQL ABSTRACTIONS
    #
    #
    def _run_sql(self, query_str):

        return pd.read_sql_query(query_str, self._conn)

    def get_train(self):

        query_str = """
        SELECT r.date_time, s.sentence, i.*
        FROM sentences s
        INNER JOIN reviews r 
        ON s.review_id = r.id
        INNER JOIN issues i
        ON s.id = i.sentence_id
        WHERE i.predicted = 0
        """
        return self._run_sql(query_str=query_str)

    def get_not_predicted(self):

        query_str = """
        SELECT r.date_time, s.review_pos, r.id, s.sentence 
        FROM sentences s
        INNER JOIN reviews r
        ON s.review_id = r.id
        WHERE NOT EXISTS(
            SELECT i.id 
            FROM issues i 
            WHERE i.sentence_id = s.id)
        """
        return self._run_sql(query_str)

    def select_from_date(self, start_date, end_date=None):

        assert type(start_date) in [date, datetime, arrow], "start date must be a date or datetime object"
        start_date = arrow.get(start_date)
        query_str = """
            SELECT s.sentence, r.date_time, i.*
            FROM sentences s
            INNER JOIN reviews r
            ON r.id = s.review_id
            LEFT JOIN issues i
            ON s.id = i.sentence_id"""
        if not end_date:
            query_str += """
            WHERE r.date_time > '{}'
            """.format(start_date.format("YYYY-MM-DD HH:mm:SS.000000"))
            return self._run_sql(query_str)
        else:
            assert type(end_date) in [date, datetime, arrow], "end date must be a date or datetime object"
            end_date = arrow.get(end_date)
            query_str += """
            WHERE r.date_time > '{}' AND r.date_time < '{}'
            """.format(start_date.format("YYYY-MM-DD HH:mm:SS.000000"), end_date.format("YYYY-MM-DD HH:mm:SS.000000"))
            return self._run_sql(query_str)

    def get_all_text(self):

        query_str = """
        SELECT s.sentence
        FROM sentences s
        """

        return self._run_sql(query_str)
