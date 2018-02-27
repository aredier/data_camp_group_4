from scrapy import Spider
from scrapy import Request
import regex as re

from ..loaders import ReviewLoader, SourceLoader, ModelLoader
from ..items import Review, Source, Model


class BestBuySpider(Spider):

    name = "best_buy_spider"

    # Custom params
    follow_pages = False

    urls = ["https://www.bestbuy.com/site/iphone/iphone-x/pcmcat1505326434742.c?id=pcmcat1505326434742"
            ]

    def start_requests(self):

        for url in self.urls:
            yield Request(
                url=url,
                callback=self.parse_request,
                meta={"name": "best_buy",
                      "url": "https://www.bestbuy.com/"}
            )

    def parse_request(self, response):
        PHONE_LINK_SELECTOR = ".list-items .list-item-postcard .sku-title a::attr(href)"
        for phone_link in response.css(PHONE_LINK_SELECTOR):
            yield response.follow(phone_link.extract(), callback=self.parse_phone_page, meta=response.meta)

    def parse_phone_page(self, response):
        REVIEWS_SELECTOR = "#reviews-content .see-all-reviews-button-container a::attr(href)"
        reviews_link = response.css(REVIEWS_SELECTOR).extract_first()
        yield response.follow(reviews_link, callback=self.parse_meta, meta=response.meta)

    def parse_meta(self, response):

        meta = {}
        source_loader = SourceLoader(item=Source(), response=response)
        source_loader.add_value("name", response.meta["name"])
        source_loader.add_value("url", response.meta["url"])
        source = source_loader.load_item()
        meta["source"] = source
        yield meta["source"]

        model_loader = ModelLoader(item=Model(), response=response)
        brand, name, memory_size = self._get_model_info(response.url)
        model_loader.add_value("brand", brand)
        model_loader.add_value("name", name)
        model_loader.add_value("memory_size", memory_size)
        model = model_loader.load_item()
        meta["model"] = model
        yield model

        yield Request(url=response.url, callback=self.parse_reviews, meta=meta, dont_filter=True)

    def parse_reviews(self, response):

        source = response.meta["source"]
        model = response.meta["model"]
        review_section = response.css(".reviews-content-wrapper")

        REVIEW_GROUP_SELECTOR = "ul li.review-item"

        for review_group in review_section.css(REVIEW_GROUP_SELECTOR):

            REVIEW_SELECTOR = ".review-wrapper .review-content p::text"
            REVIEW_DATE_SELECTOR = ".review-wrapper .review-date span::text"
            SOURCE_REVIEW_ID_SELECTOR = "li.review-item::attr(data-topic-id)"
            RATING_SELECTOR = ".reviewer-score::text"
            HELPED_SELECTOR = ".feedback-display a:nth-child(1) span::text"

            review_loader = ReviewLoader(item=Review(), response=response)

            review_text = review_group.css(REVIEW_SELECTOR).extract_first()
            review_date = review_group.css(REVIEW_DATE_SELECTOR).extract_first()
            source_review_id = review_group.css(SOURCE_REVIEW_ID_SELECTOR).extract_first()
            try :
                review_rating = int(review_group.css(RATING_SELECTOR).extract_first()) / 5
                review_loader.add_value("rating", review_rating)
            except TypeError:
                pass
            try :
                review_helped = int(review_group.css(HELPED_SELECTOR).extract_first())
                review_loader.add_value("helped", review_helped)
            except TypeError:
                pass

            review_loader.add_value("review", review_text)
            review_loader.add_value("date_time", review_date)
            review_loader.add_value("source_name", source["name"])
            review_loader.add_value("source_url", source["url"])
            review_loader.add_value("model_name", model["name"])
            review_loader.add_value("model_brand", model["brand"])
            review_loader.add_value("model_memory_size", model["memory_size"])
            review_loader.add_value("source_review_id", source_review_id)

            review = review_loader.load_item()

            yield review

        NEXT_PAGE_SELECTOR = "ul.pagination li.active+li a::attr(href)"

        url_request = response.css(NEXT_PAGE_SELECTOR)
        if url_request:
            url = url_request.extract_first()

            if self.follow_pages:
                yield response.follow(url, callback=self.parse_reviews, meta=response.meta)

    @staticmethod
    def _get_model_info(url):
        model_string = url.split("/")[-2]
        model_re = re.match(r"([a-z]+)-([a-z0-9\-\+]+)-([0-9]+)gb-", model_string)
        brand = model_re.captures(1)[0]
        name = model_re.captures(2)[0]
        memory_size = str(model_re.captures(3)[0])
        return brand, name, memory_size