# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field


class Review(Item):
    """scrapy data sctructure for the reviews"""

    review = Field()
    date_time = Field()
    rating = Field()
    helped = Field()

    source_name = Field()
    source_url = Field()
    source_review_id = Field()
    model_name = Field()
    model_brand = Field()
    model_memory_size = Field()


class Source(Item):
    """scrapy data structure to handel sources"""

    name = Field()
    url = Field()


class Model(Item):

    name = Field()
    brand = Field()
    memory_size = Field()
