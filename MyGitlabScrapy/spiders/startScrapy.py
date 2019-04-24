from scrapy import cmdline
from scrapy.cmdline import execute

array_lbh = ['https://gitlab.com/postmarketOS/pmbootstrap','https://gitlab.com/openconnect/ocserv','https://gitlab.com/tom79/mastalab','https://gitlab.com/Mr_Goldberg/goldberg_emulator','https://gitlab.com/sleepyhead/sleepyhead-code','https://gitlab.com/fatihacet/gitlab-vscode-extension','https://gitlab.com/mojo42/Jirafeau','https://gitlab.com/painlessMesh/painlessMesh','https://gitlab.com/flectra-hq/flectra','https://gitlab.com/Mactroll/NoMAD','https://gitlab.com/git-latexdiff/git-latexdiff','https://gitlab.com/gitlab-org/gitlab-shell','https://gitlab.com/gitlab-org/gitlab-runner-docker-cleanup','https://gitlab.com/sortix/sortix','https://gitlab.com/gableroux/unity3d-gitlab-ci-example','https://gitlab.com/cunidev/gestures','https://gitlab.com/sequoia-pgp/sequoia','https://gitlab.com/postmill/Postmill','https://gitlab.com/pdfgrep/pdfgrep','https://gitlab.com/dessalines/torrents.csv','https://gitlab.com/angelkyo/w10-digitallicense','https://gitlab.com/MasterPassword/MasterPassword','https://gitlab.com/NebulousLabs/Sia','https://gitlab.com/procps-ng/procps','https://gitlab.com/eidheim/Simple-Web-Server']
array_lwg = ['https://gitlab.com/meno/dropzone','https://gitlab.com/pycqa/flake8','https://gitlab.com/commento/commento','https://gitlab.com/leanlabsio/kanban','https://gitlab.com/staltz/manyverse','https://gitlab.com/swisspost/evoting-solution','https://gitlab.com/mayan-edms/mayan-edms','https://gitlab.com/recalbox/recalbox','https://gitlab.com/gitlab-org/gitlab-ci','https://gitlab.com/gitlab-com/runbooks','https://gitlab.com/esr/open-adventure','https://gitlab.com/OpenMW/openmw','https://gitlab.com/gitlab-org/gitlab-mattermost','https://gitlab.com/fdroid/fdroidserver','https://gitlab.com/meltano/meltano','https://gitlab.com/xhang/gitlab','https://gitlab.com/somerobots/Trident','https://gitlab.com/esr/upside','https://gitlab.com/mailman/mailman','https://gitlab.com/o9000/tint2','https://gitlab.com/pages/hugo','https://gitlab.com/tezos/tezos','https://gitlab.com/technomancy/bussard','https://gitlab.com/gitlab-org/gitlab-design','https://gitlab.com/Alvin9999/free']
array_skw = ['https://gitlab.com/Isleward/isleward','https://gitlab.com/muttmua/mutt','https://gitlab.com/tildes/tildes','https://gitlab.com/mbunkus/mkvtoolnix','https://gitlab.com/pages/jekyll','https://gitlab.com/ase/ase','https://gitlab.com/stavros/Spamnesty','https://gitlab.com/gitlab-com/gl-infra/infrastructure','https://gitlab.com/gitlab-com/support-forum','https://gitlab.com/AuroraOSS/AuroraStore','https://gitlab.com/charts/gitlab','https://gitlab.com/prismosuite/prismo','https://gitlab.com/gitlab-org/gitaly','https://gitlab.com/gnutls/gnutls','https://gitlab.com/antora/antora','https://gitlab.com/pages/plain-html','https://gitlab.com/mcmfb/intro_x86-64','https://gitlab.com/kivymd/KivyMD','https://gitlab.com/oeffi/oeffi','https://gitlab.com/embeddable-common-lisp/ecl','https://gitlab.com/kornelski/babel-preset-php','https://gitlab.com/higan/higan','https://gitlab.com/Remmina/Remmina','https://gitlab.com/xonotic/xonotic','https://gitlab.com/ganttlab/ganttlab-live']
array_ld = ['https://gitlab.com/postmarketOS/pmbootstrap','https://gitlab.com/openconnect/ocserv','https://gitlab.com/tom79/mastalab','https://gitlab.com/Mr_Goldberg/goldberg_emulator','https://gitlab.com/sleepyhead/sleepyhead-code','https://gitlab.com/fatihacet/gitlab-vscode-extension','https://gitlab.com/mojo42/Jirafeau','https://gitlab.com/painlessMesh/painlessMesh','https://gitlab.com/flectra-hq/flectra','https://gitlab.com/Mactroll/NoMAD','https://gitlab.com/git-latexdiff/git-latexdiff','https://gitlab.com/gitlab-org/gitlab-shell','https://gitlab.com/gitlab-org/gitlab-runner-docker-cleanup','https://gitlab.com/sortix/sortix','https://gitlab.com/gableroux/unity3d-gitlab-ci-example','https://gitlab.com/cunidev/gestures','https://gitlab.com/sequoia-pgp/sequoia','https://gitlab.com/postmill/Postmill','https://gitlab.com/pdfgrep/pdfgrep','https://gitlab.com/dessalines/torrents.csv','https://gitlab.com/angelkyo/w10-digitallicense','https://gitlab.com/MasterPassword/MasterPassword','https://gitlab.com/NebulousLabs/Sia','https://gitlab.com/procps-ng/procps','https://gitlab.com/eidheim/Simple-Web-Server']

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


