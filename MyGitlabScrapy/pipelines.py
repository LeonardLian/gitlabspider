# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import codecs
import json

class MygitlabscrapyPipeline(object):
    def __init__(self):
        self.file = codecs.open(filename="project_commit.json", mode='w+')

    def process_item(self, item, spider):

        # filename = (item["commit_href"].split('/'))[3]+'_'+(item["commit_href"].split('/'))[4]+'.json'

        filename = item["project_name"]+'_commitInfo.json'

        self.file = codecs.open(filename=filename, mode='a+')
        data_str = json.dumps(dict(item), ensure_ascii=False)+",\n"
        print "---------"
        print data_str
        self.file.write(data_str)
        return item

