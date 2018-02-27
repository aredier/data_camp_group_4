import pandas as pd

import sqlite3

from tqdm import tqdm

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

from datetime import date

from .models import Base, Sentences, Reviews


SQLITE_REL_PATH = 'data/review_data.db'


def add_issue(label, path_to_database):
    """adds an issue colum to the issues table """
    conn = sqlite3.connect(path_to_database)
    try:
        conn.execute("ALTER TABLE issues ADD COLUMN {} INTEGER".format(label))
    except sqlite3.OperationalError:
        pass


def _insert_unlabeled(session, row, insert_date):
    """inserts an issue from the unlabled dataframe"""
    review = Reviews(date_time=insert_date)
    session.add(review)
    session.commit()
    sentence = Sentences(sentence=row["text"], review_id=review.id, review_pos=0)
    session.add(sentence)
    session.commit()


def _insert_labeled(row, session, IssueClass, insert_date):
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


def build_data_base(path_to_database, labeled=None, unlabeled=None):
    """builds the data base and populates it with reviews form labeled and unlabeled data frame if specified"""
    tqdm.pandas()

    engine = create_engine("sqlite:///"+path_to_database, echo=False)
    Base.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    insert_date = date(2018, 1, 1)

    if labeled:
        print("importing labeled data")
        labeled = pd.read_csv(labeled)
        for label in labeled:
            if label == "text":
                continue
            add_issue(label, path_to_database)
        session = session_maker()
        Base2 = automap_base()
        Base2.prepare(engine, reflect=True)
        Issue = Base2.classes.issues
        labeled.progress_apply(lambda row: _insert_labeled(row, session, Issue, insert_date), axis=1)
        session.close()

    if unlabeled:
        print("importing unalabeled data")
        unlabeled = pd.read_csv(unlabeled)
        session = session_maker()
        unlabeled.progress_apply(lambda row: _insert_unlabeled(session, row, insert_date), axis=1)
        session.close()


if __name__ == "__main__":
    build_data_base(SQLITE_REL_PATH)
