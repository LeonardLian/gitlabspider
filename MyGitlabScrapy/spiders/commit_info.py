import scrapy
from MyGitlabScrapy.items import CommitInfoItem
import json
import os

class MySpider(scrapy.Spider):
    name = 'commit_info'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36',
    }

    def __init__(self, filePath=None, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        # self.filePath = filePath

    def start_requests(self):

        start_urls = []
        # file = open(self.filePath, "rb")

        files = os.listdir(os.getcwd())

        for json_file in files:
            if json_file.split(".")[-1] == "json" and json_file.split(".")[0] != "project_commit":
                print json_file
                file_info = open(json_file, "rb")
                data = json.load(file_info)
                for item in data:
                    # print item["commit_href"]
                    start_urls.append(item["commit_href"])

            else:
                pass

        for url in start_urls:
            print url
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        commit_info = CommitInfoItem()

        page = (response.url.split("/"))[3]+"_"+(response.url.split("/"))[4]

        commit_info['project_name'] = page

        commit_info['commit_id'] = response.xpath( '//div[@class="header-main-content"]/button[@class="btn btn-clipboard btn-transparent"]/@data-clipboard-text').extract()
        commit_info['commit_time'] = response.xpath('//div[@class="header-main-content"]/time/@datetime').extract()
        commit_info['author_name'] = response.xpath('//span[@class="commit-author-name"]/text()').extract()

        commit_info['commit_title'] = response.xpath('//h3[@class="commit-title"]/text()').extract()
        commit_info['commit_description'] = response.xpath('//pre[@class="commit-description"]/text()').extract()

        commit_info['changed_file_num'] = response.xpath('//button[@class="diff-stats-summary-toggler js-diff-stats-dropdown"]/text()').extract()
        commit_info['changed_file'] = response.xpath( '//div[@class="file-header-content"]/a/strong[@class="file-title-name has-tooltip"]/text()').extract()

        commit_info['additions_num'] = response.xpath('//span[@class="diff-stats-additions-deletions-expanded"]/strong[@class="cgreen"]/text()').extract()
        commit_info['deletions_num'] = response.xpath('//span[@class="diff-stats-additions-deletions-expanded"]/strong[@class="cred"]/text()').extract()
        # print commit_info
        yield commit_info
