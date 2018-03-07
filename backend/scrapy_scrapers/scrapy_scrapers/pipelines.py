# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from nltk import sent_tokenize
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from ...models import Sentences, Sources, Models, Reviews
from .items import Review, Source, Model
from .settings import SQLITE_REL_PATH


class ScrapyScrapersPipeline(object):

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        path_to_base = settings["path"]
        last_update = settings["date_time"]

        return cls(path=path_to_base, date_time=last_update)

    def __init__(self, path=None, date_time=None):
        """create pipeline"""
        if path:
            self.path_to_base = path
        else:
            self.path_to_base = SQLITE_REL_PATH
        if date_time:
            self.date_time = date_time
        self.engine = create_engine(self._db_url)
        self.session = sessionmaker(bind=self.engine)
        self.logger = None

    def open_spider(self, spider):
        """
        function that sets the logger


        :type spider: Scrapy.Spider
        """
        self.logger = spider.logger

    def process_item(self, item, spider):
        session = self.session()

        try:
            if isinstance(item, Review):
                self._review_insert(session, item)
            elif isinstance(item, Source):
                self._source_insert(session, item)
            elif isinstance(item, Model):
                self._model_insert(session, item)
            session.commit()
        except Exception as e:
            self.logger.error("error when inserting into db : {}".format(e))

        finally:
            session.close()

    def _generic_insert(self, session, model, item):
        if not model or not item:
            return
        self.logger.info("Inserting entry of type \"{}\" in DB".format(model.__tablename__))
        item = model(**item)
        session.add(item)
        session.commit()
        return item

    def _source_insert(self, session, item):
        if not item:
            return
        if session.query(Sources).filter(Sources.name == item["name"], Sources.url == item["url"]).first():
            return
        return self._generic_insert(session, Sources, item)

    def _model_insert(self, session, item):
        if not item:
            return
        test = session.query(Models).filter(Models.name == item["name"],
                                            Models.brand == item["brand"],
                                            item["memory_size"] == Models.memory_size).first()
        if test:
            return
        return self._generic_insert(session, Models, item)

    def _sentences_insert(self, session, review_id, review_text):
        for i, sent in enumerate(sent_tokenize(review_text)):
            self._generic_insert(session, Sentences, {
                "sentence": sent,
                "review_id": review_id,
                "review_pos": i})

    def _review_insert(self, session, item):
        if not item:
            return

        # checking we are in date range
        try:
            if self.date_time >= item["date_time"]:
                self.logger.info("passing review before last update date")
                return
        except KeyError:
            return

        # retrieving source id
        source_name = item.pop("source_name")
        source_url = item.pop("source_url")
        source_request = session.query(Sources).filter(Sources.name == source_name, Sources.url == source_url).first()
        if not source_request:
            source = self._source_insert(session, {
                "name": source_name,
                "url": source_url})
            source_id = source.id
        else:
            source_id = source_request.id

        # retrieving model id
        model_name = item.pop("model_name")
        model_brand = item.pop("model_brand")
        try:
            model_memory_size = item.pop("model_memory_size")
        except KeyError:
            model_memory_size = None
        model_request = session.query(Models).filter(model_name == Models.name,
                                                     model_brand == Models.brand,
                                                     model_memory_size == Models.memory_size).first()
        if not model_request:
            model = self._model_insert(session, {
                "name": model_name,
                "brand": model_brand,
                "memory_size": model_memory_size
            })
            model_id = model.id
        else:
            model_id = model_request.id

        # checking if the review is not aleready in the data base
        try:
            soure_review_id = item["source_review_id"]
            review_request = session.query(Reviews).filter(Reviews.source_review_id == soure_review_id,
                                                           Reviews.source_id == source_id).first()
            if review_request:
                    return None
        except KeyError:
            pass

        try:
            review_params = {"date_time": item["date_time"],
                             "source_id": source_id,
                             "model_id": model_id}
        except KeyError:
            return None

        try:
            review_params["rating"] = item["rating"]
        except KeyError:
            pass
        try:
            review_params["helped"] = item["helped"]
        except KeyError:
            pass

        review = self._generic_insert(session, Reviews, review_params)

        self._sentences_insert(session, review.id, item["review"])

    @property
    def _db_url(self):
        return "sqlite:///" + self.path_to_base
