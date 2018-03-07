from scrapy import Spider
from scrapy import Request

from ..loaders import ReviewLoader, SourceLoader
from ..items import Review, Source

class AndroidCentralSpider(Spider):

    name = "android_central_spider"
    allowed_domains = ["https://www.androidcentral.com"]

    # Custom params
    follow_pages = False

    urls = ["https://www.androidcentral.com/samsung-galaxy-note-8-review-three-months"]

    def start_requests(self):


        for url in self.urls:
            yield Request(
                url=url,
                callback=self.parse_page,
                meta={"source_name" : "android_central",
                      "source_url" : url}
            )


    def parse_page(self, response):

        source_name = response.meta["source_name"]
        source_url = response.meta["source_url"]

        source_loader = SourceLoader(item=Source(), response=response)
        source_loader.add_value("name", source_name)
        source_loader.add_value("url", source_url)
        yield source_loader.load_item()

        comments = response.css(".ajax-comment-wrapper")

        for comment in comments :

            COMMENT_SELECTOR = 'article div p ::text'

            review_text = comment.css(COMMENT_SELECTOR).extract_first()

            review_loader = ReviewLoader(item=Review(), response=response)
            review_loader.add_value("review", review_text)
            review_loader.add_value("source_name", source_name)
            review_loader.add_value("source_url", source_url)

            review = review_loader.load_item()

            yield review

        NEXT_PAGE_SELECTOR = ".next"
