import scrapy
from MyGitlabScrapy.items import ProjectCommitItem
import sys
import re


class MySpider(scrapy.Spider):
    name = 'project_commit'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36',
    }

    def __init__(self, baseURL=None, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.baseURL = baseURL

    def start_requests(self):
        URLstr = '?limit=40&offset='
        # start_urls = array_ld

        start_urls = ["https://gitlab.com/esr/open-adventure/commits/master"]

        template_url = []
        for base in start_urls:
            template_url.append(base)

        for baseURL in template_url:
            offset = 40
            while offset < 60000:
                # while offset < 1200:
                offset = offset + 40
                url = baseURL + URLstr + str(offset)
                print "add "+url
                start_urls.append(url)

        for url in start_urls:
            print url
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        project_commit = ProjectCommitItem()
        project_commits = response.xpath('//li[@class="commit flex-row js-toggle-container"]')

        for commit in project_commits:
            p1 = re.compile('.*Commit: .*')
            project_commit['commit_href'] = "https://gitlab.com" + commit.xpath('.//div[@class="commit-detail flex-list"]/div[@class ="commit-content qa-commit-content"]/a/@href').extract()[0]
            build_result_exist= commit.xpath('.//div[@class="commit-detail flex-list"]/div[@class ="commit-actions flex-row"]').extract()[0]
            exist_or_not = re.search(p1, build_result_exist)

            if exist_or_not:

                # project_commit['build_result'] = commit.xpath('.//div[@class="commit-detail flex-list"]/div[@class ="commit-content qa-commit-content"]/div[@class ="d-block d-sm-none"]/a/@title').extract()[0]

                project_commit['build_result'] = commit.xpath('.//div[@class="commit-detail flex-list"]/div[@class ="commit-actions flex-row"]/a/@title').extract()[0]
            else:
                project_commit['build_result'] = ""
            print project_commit
            yield project_commit


