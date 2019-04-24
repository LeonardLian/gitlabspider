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



str_ld = [
          'https://gitlab.com/postmarketOS/pmbootstrap',
          'https://gitlab.com/openconnect/ocserv',
          'https://gitlab.com/tom79/mastalab',
          'https://gitlab.com/Mr_Goldberg/goldberg_emulator',
          'https://gitlab.com/sleepyhead/sleepyhead-code',
          'https://gitlab.com/fatihacet/gitlab-vscode-extension',
          'https://gitlab.com/mojo42/Jirafeau',
          'https://gitlab.com/painlessMesh/painlessMesh',
          'https://gitlab.com/flectra-hq/flectra',
          'https://gitlab.com/Mactroll/NoMAD',
          'https://gitlab.com/git-latexdiff/git-latexdiff',
          'https://gitlab.com/gitlab-org/gitlab-shell',
          'https://gitlab.com/gitlab-org/gitlab-runner-docker-cleanup',
          'https://gitlab.com/sortix/sortix',
          'https://gitlab.com/gableroux/unity3d-gitlab-ci-example',
          'https://gitlab.com/cunidev/gestures',
          'https://gitlab.com/sequoia-pgp/sequoia',
          'https://gitlab.com/postmill/Postmill',
          'https://gitlab.com/pdfgrep/pdfgrep',
          'https://gitlab.com/dessalines/torrents.csv',
          'https://gitlab.com/angelkyo/w10-digitallicense',
          'https://gitlab.com/MasterPassword/MasterPassword',
          'https://gitlab.com/NebulousLabs/Sia',
          'https://gitlab.com/procps-ng/procps',
          'https://gitlab.com/eidheim/Simple-Web-Server']