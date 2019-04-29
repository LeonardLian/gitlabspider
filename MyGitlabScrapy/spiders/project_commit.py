import scrapy
from MyGitlabScrapy.items import ProjectCommitItem
import sys
import re


array_ld = ['https://gitlab.com/postmarketOS/pmbootstrap','https://gitlab.com/openconnect/ocserv','https://gitlab.com/tom79/mastalab','https://gitlab.com/Mr_Goldberg/goldberg_emulator','https://gitlab.com/sleepyhead/sleepyhead-code','https://gitlab.com/fatihacet/gitlab-vscode-extension','https://gitlab.com/mojo42/Jirafeau','https://gitlab.com/painlessMesh/painlessMesh','https://gitlab.com/flectra-hq/flectra','https://gitlab.com/Mactroll/NoMAD','https://gitlab.com/git-latexdiff/git-latexdiff','https://gitlab.com/gitlab-org/gitlab-shell','https://gitlab.com/gitlab-org/gitlab-runner-docker-cleanup','https://gitlab.com/sortix/sortix','https://gitlab.com/gableroux/unity3d-gitlab-ci-example','https://gitlab.com/cunidev/gestures','https://gitlab.com/sequoia-pgp/sequoia','https://gitlab.com/postmill/Postmill','https://gitlab.com/pdfgrep/pdfgrep','https://gitlab.com/dessalines/torrents.csv','https://gitlab.com/angelkyo/w10-digitallicense','https://gitlab.com/MasterPassword/MasterPassword','https://gitlab.com/NebulousLabs/Sia','https://gitlab.com/procps-ng/procps','https://gitlab.com/eidheim/Simple-Web-Server']
array_lbh = ['https://gitlab.com/gitlab-org/gitlab-ce','https://gitlab.com/gitlab-org/gitlab-runner','https://gitlab.com/fdroid/fdroidclient','https://gitlab.com/inkscape/inkscape','https://gitlab.com/rosarior/awesome-django','https://gitlab.com/gitlab-org/gitlab-ee','https://gitlab.com/gnachman/iterm2','https://gitlab.com/Commit451/LabCoat','https://gitlab.com/gitlab-org/gitter/webapp','https://gitlab.com/pgjones/quart','https://gitlab.com/gitlab-com/www-gitlab-com','https://gitlab.com/NickBusey/HomelabOS','https://gitlab.com/fusionjack/adhell3','https://gitlab.com/fdroid/fdroiddata','https://gitlab.com/doctorj/interview-questions','https://gitlab.com/gitlab-org/omnibus-gitlab','https://gitlab.com/cryptsetup/cryptsetup','https://gitlab.com/tortoisegit/tortoisegit','https://gitlab.com/mildlyparallel/pscircle','https://gitlab.com/gitlab-org/gitlab-development-kit','https://gitlab.com/graphviz/graphviz','https://gitlab.com/jstpcs/lnxpcs','https://gitlab.com/Rich-Harris/buble','https://gitlab.com/terrakok/gitlab-client','https://gitlab.com/bitfireAT/davx5-ose']

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

        start_urls = [
                      'https://gitlab.com/gitlab-org/gitlab-runner/commits/master',
                      'https://gitlab.com/bitfireAT/davx5-ose/commits/master-ose'
                      ]

        template_url = []
        for base in start_urls:
            template_url.append(base)

        for baseURL in template_url:
            offset = 40
            while offset < 60000:
                # while offset < 1200:
                offset = offset + 40
                url = baseURL + URLstr + str(offset)
                # print "add "+url
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


