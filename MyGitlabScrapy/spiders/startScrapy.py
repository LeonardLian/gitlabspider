from scrapy import cmdline
from scrapy.cmdline import execute


def startScrapy():

    baseURL = "https://gitlab.com/fdroid/fdroidclient/commits/master";
    project_commit_str = "scrapy crawl project_commit -a baseURL="+baseURL+" -o project_commit.json -s LOG_FILE=project_commit.log"
    print project_commit_str
    cmdline.execute(project_commit_str.split())

    filePath = "MyGitlabScrapy/spiders/project_commit.json"
    commit_info_str = "scrapy crawl commit_info -a filePath="+filePath+" -o commit_info.json -s LOG_FILE=commit_info.log"
    cmdline.execute(commit_info_str.split())

def main():
    startScrapy()

if __name__ == '__main__':
    main()
