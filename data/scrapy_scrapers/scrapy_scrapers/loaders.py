# -*- coding: utf-8 -*-

from datetime import date
import arrow

from scrapy.loader import ItemLoader
from scrapy.loader.processors import TakeFirst, MapCompose


def _parse_date(date):
    try:
        return arrow.get(date, "MMMM DD, YYYY").datetime.replace(tzinfo=None)
    except:
        return None


class ReviewLoader(ItemLoader):

    source_name_in = MapCompose(str)
    source_url_in = MapCompose(str)
    review_in = MapCompose(str)
    date_time_in = MapCompose(_parse_date)
    rating_in = MapCompose(float)
    helped_in = MapCompose(int)
    source_review_id_in = MapCompose(str)
    model_name = MapCompose(str)
    model_brand = MapCompose(str)
    model_memory_size = MapCompose(int)

    default_output_processor = TakeFirst()


class SourceLoader(ItemLoader):

    name_in = MapCompose(str)
    url_in = MapCompose(str)

    default_output_processor = TakeFirst()


class ModelLoader(ItemLoader):

    name_in = MapCompose(str)
    brand_in = MapCompose(str)
    memory_size_in = MapCompose(int)

    default_output_processor = TakeFirst()


