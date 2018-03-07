from scrapy import Spider
from scrapy import Request

from ..loaders import ReviewLoader, SourceLoader
from ..items import Review, Source

class AmazonSpider(Spider):

    name = "amazon_spider"
    allowed_domains = ["https://www.amazon.com/Apple-iPhone-GSM-Unlocked-5-8/product-reviews/B075QMZH2L/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"]

    # Custom params
    follow_pages = False

    urls = ["http://www.amazon.com"]

    def start_requests(self):


        for url in self.urls:
            yield Request(
                url=url,
                callback=self.parse_page,
                meta={"source_name" : "amazon",
                      "source_url" : url}
            )


    def parse_page(self, response):

        source_name = response.meta["source_name"]
        source_url = response.meta["source_url"]

        review_section = response.css(".view-point .a-fixed-right-grid-inner .a-col-left").extract_first()

        REVIEW_GROUP_SELECTOR = ".review"

        for review_group in review_section.css(REVIEW_GROUP_SELECTOR):

            REVIEW_SELECTOR = "div.a-row:nth-child(4) span.review-text::text"

            review_text = review_group.css(REVIEW_SELECTOR).extract_first()

            review_loader = ReviewLoader(item=Review(), response=response)
            review_loader.add_value("review", review_text)
            review_loader.add_value("source_name", source_name)
            review_loader.add_value("source_url", source_url)

            review = review_loader.load_item()

            yield review
