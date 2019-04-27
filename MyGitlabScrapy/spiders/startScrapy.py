from scrapy import cmdline
from scrapy.cmdline import execute


def startScrapy():
    # project_commit_str = "scrapy crawl project_commit -s LOG_FILE=project_commit.log"
    # print project_commit_str
    # cmdline.execute(project_commit_str.split())

    commit_info_str = "scrapy crawl commit_info -s LOG_FILE=commit_info.log"
    print commit_info_str
    cmdline.execute(commit_info_str.split())

def main():
    startScrapy()

if __name__ == '__main__':
    main()


