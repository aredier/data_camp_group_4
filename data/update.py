"""" class that handeles the updates of the database"""
from datetime import datetime

from scrapy.crawler import CrawlerProcess
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Reviews
from .scrapy_scrapers.scrapy_scrapers.spiders.best_buy_spider import BestBuySpider


def update(path_to_base, date=None):
    if not date:
        engine = create_engine("sqlite:///" + path_to_base, echo=False)
        session_maker = sessionmaker(bind=engine)
        session = session_maker()

        date = session.query(Reviews).distinct(Reviews.date_time).order_by(Reviews.date_time.desc()).first()
        if not date :
            date = datetime(2017, 1, 1).replace(tzinfo=None)
        else:
            date = date.date_time.replace(tzinfo=None)


    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:57.0) Gecko/20100101 Firefox/57.0',
        "path": path_to_base,
        "date_time" : date,
        "LOG_LEVEL" : "INFO",
        "ITEM_PIPELINES": {
            'group_4_backend.data.scrapy_scrapers.scrapy_scrapers.pipelines.ScrapyScrapersPipeline': 300,
        },
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 32,
        "CONCURRENT_REQUESTS_PER_DOMAIN ": 32,
        "AUTOTHROTTLE_ENABLED": False,
        "LOG_LEVEL": "INFO"
    })

    process.crawl(BestBuySpider)
    process.start()