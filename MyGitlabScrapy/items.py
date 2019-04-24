# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ProjectItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    # 项目ID
    project_id = scrapy.Field()




    # 项目大小
    project_size = scrapy.Field()

    # 项目star
    project_star = scrapy.Field()

    # 项目commit
    project_commit = scrapy.Field()

    # 项目branch数
    project_branch = scrapy.Field()
    pass


class CommitInfoItem(scrapy.Item):
    commit_id = scrapy.Field()
    commit_time = scrapy.Field()
    author_name = scrapy.Field()
    commit_title = scrapy.Field()
    commit_description = scrapy.Field()
    changed_file_num = scrapy.Field()
    changed_file = scrapy.Field()
    additions_num = scrapy.Field()
    deletions_num = scrapy.Field()
    pass

class ProjectCommitItem(scrapy.Item):
    commit_href = scrapy.Field()
    build_result = scrapy.Field()
    pass