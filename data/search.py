""" module that allows to query the database and retrun appropriate data frames"""

import pandas as pd
import sqlite3
from datetime import date, datetime
import arrow

def get_train(path_to_db):
    conn = sqlite3.connect(path_to_db)

    train = pd.read_sql_query("""
    SELECT r.date_time, r.id as review_id, s.sentence, i.*
    FROM sentences s
    INNER JOIN reviews r 
    ON s.review_id = r.id
    INNER JOIN issues i
    ON s.id = i.sentence_id
    WHERE i.predicted = 0
    """, conn)
    return train

def get_not_predicted(path_to_db):
    conn = sqlite3.connect(path_to_db)

    not_predicted = pd.read_sql_query("""
    SELECT r.date_time, s.review_pos, r.id, s.sentence 
    FROM sentences s
    INNER JOIN reviews r
    ON s.review_id = r.id
    WHERE NOT EXISTS(
        SELECT i.id 
        FROM issues i 
        WHERE i.sentence_id = s.id)
    """, conn)
    return not_predicted

def select_from_date(path_to_db, start_date, end_date=None):
    conn = sqlite3.connect(path_to_db)
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
        query_sentences = pd.read_sql_query(query_str + """
        WHERE r.date_time > '{}'
        """.format(start_date.format("YYYY-MM-DD HH:mm:SS.000000")), conn)#change review id by sentence_id when base updated in test
    else:
        print("got_there")
        assert type(end_date) in [date, datetime, arrow], "end date must be a date or datetime object"
        end_date = arrow.get(end_date)
        query_sentences = pd.read_sql_query(query_str + """
        WHERE r.date_time > '{}' AND r.date_time < '{}'
        """.format(start_date.format("YYYY-MM-DD HH:mm:SS.000000"), end_date.format("YYYY-MM-DD HH:mm:SS.000000")), conn)#change review id by sentence_id when base updated in test


    return query_sentences

