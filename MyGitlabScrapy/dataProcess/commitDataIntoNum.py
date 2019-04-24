# encoding=utf8
import sys
import json
import re
import datetime
import time
import pytz
from collections import deque
import numpy as np
import pandas as pd


def openJsonFile(filePath):
    file_info = open(filePath, "rb")
    info_data = json.load(file_info)
    return info_data


# 将时间整理为数值形式
def utc_to_local(utc_time_str, utc_format='%Y-%m-%dT%H:%M:%SZ'):
    local_tz = pytz.timezone('Asia/Chongqing')
    local_format = "%Y-%m-%d %H:%M"
    utc_dt = datetime.datetime.strptime(utc_time_str, utc_format)
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    time_str = local_dt.strftime(local_format)
    return int(time.mktime(time.strptime(time_str, local_format)))


# 将构建结果和链接  与  commit各项信息合并
def commits_info_merge(filePath1, filePath2, filePath3):
    info_data = openJsonFile(filePath1)
    result_data = openJsonFile(filePath2)

    p1 = re.compile('[^/]+(?!.*/)')

    commit_info_merge = []
    for result_item in result_data:
        search_id = re.search(p1, result_item["commit_href"]).group(0)
        for info_item in info_data:
            commit_id = info_item["commit_id"]
            if search_id == commit_id[0]:

                info_item["commit_href"] = result_item["commit_href"]
                info_item["build_result"] = result_item["build_result"]

                info_item["commit_id"] = info_item["commit_id"][0]
                info_item["additions_num"] = info_item["additions_num"][0]
                info_item["commit_title"] = info_item["commit_title"][0]
                info_item["author_name"] = info_item["author_name"][0]
                info_item["changed_file_num"] = info_item["changed_file_num"][0]
                info_item["deletions_num"] = info_item["deletions_num"][0]
                if info_item["commit_description"]:
                    info_item["commit_description"] = info_item["commit_description"][0]
                else:
                    info_item["commit_description"] = ""

                info_time = info_item["commit_time"][0]
                local_tz = pytz.timezone('Asia/Chongqing')
                utc_dt = datetime.datetime.strptime(info_time, '%Y-%m-%dT%H:%M:%SZ')
                local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
                info_item["time"] = local_dt.hour
                info_item["month_day"] = local_dt.day
                info_item["weekday"] = local_dt.weekday()

                info_item["commit_time"] = str(utc_to_local(info_time))

                commit_info_merge.append(info_item)
                break

    with open(filePath3, 'w') as json_file:
        json_file.write(json.dumps(sorted(commit_info_merge, key=lambda x: x['commit_time']), indent=4))


# 获取每个contributer提交的总次数，制成新文件
def get_contributer_info(file_path1, file_path2):
    info_data = openJsonFile(file_path1)
    final_data = [{"author_name": "", "auther_commit_total": "1",
                   "build_total": 0, "build_success": 0, "build_fail": 0, "committer_recent": []}]
    p1 = re.compile('\d+')
    flag = 0
    newauthor = {"author_name": "", "auther_commit_total": "1",
                 "build_total": 0, "build_success": 0, "build_fail": 0, "committer_recent": []}

    for info_data_item in info_data:

        contributer = info_data_item["author_name"]
        build_result = str(info_data_item["build_result"])
        commit_time = int(info_data_item["commit_time"])

        for final_contributer in final_data:
            if contributer == final_contributer["author_name"]:
                final_contributer["auther_commit_total"] = str(
                    int(re.search(p1, final_contributer["auther_commit_total"]).group()) + 1).decode('utf-8')
                if build_result != "":
                    final_contributer["build_total"] += 1
                    if build_result == "Commit: failed":
                        final_contributer["build_fail"] += 1
                        queue = deque(final_contributer["committer_recent"], maxlen=5)
                        queue.append("fail")
                        length = min(len(final_contributer["committer_recent"]) + 1, 5)

                        final_contributer["committer_recent"] = []
                        i = 0
                        while i < length:
                            final_contributer["committer_recent"].append(queue.popleft())
                            i += 1

                    elif build_result == "Commit: passed":
                        final_contributer["build_success"] += 1
                        queue = deque(final_contributer["committer_recent"], maxlen=5)
                        queue.append("success")
                        length = min(len(final_contributer["committer_recent"]) + 1, 5)

                        final_contributer["committer_recent"] = []
                        i = 0
                        while i < length:
                            final_contributer["committer_recent"].append(queue.popleft())
                            i += 1
                    else:
                        pass
                flag = 0
                break
            else:
                flag = 1
        if flag == 1:
            newauthor["author_name"] = info_data_item["author_name"]
            final_data.append(newauthor)
            newauthor = {"author_name": "", "auther_commit_total": "1",
                         "build_total": 0, "build_success": 0, "build_fail": 0, "committer_recent": []}

    with open(file_path2, 'w') as json_file:
        json_file.write(json.dumps(final_data, indent=4))


# 把contributer提交的总次数与commit的信息合并
def merge_contributer_commit(filePath6, filePath7, filePath8):
    contributer_data = openJsonFile(filePath6)
    info_data = openJsonFile(filePath7)
    commit_info_merge_contributer = []

    commit_tag = 0
    for info_data_item in info_data:
        for contributer_data_item in contributer_data:
            if info_data_item["author_name"] == contributer_data_item["author_name"]:
                info_data_item["auther_commit_total"] = contributer_data_item["auther_commit_total"]
                commit_info_merge_contributer.append(info_data_item)
                break
        info_data_item["commit_tag"] = str(commit_tag).decode('utf-8')
        commit_tag = commit_tag + 1

    with open(filePath8, 'w') as json_file:
        json_file.write(json.dumps(sorted(commit_info_merge_contributer, key=lambda x: x['commit_time']), indent=4))


# 得到每次commit源文件和配置文件修改的个数
def getJavaConfigNum(info_data_item):
    java_count = 0
    rb_count = 0
    xml_count = 0
    config_count = 0
    pjava = re.compile('.*\.java.*')
    prb = re.compile('.*\.rb.*')
    pxml = re.compile('.*\.xml.*')
    pconfig = re.compile('.*\.config.*')
    for one_changed_file in info_data_item["changed_file"]:
        if re.search(pjava, one_changed_file):
            java_count = java_count + 1
        if re.search(prb, one_changed_file):
            rb_count = rb_count + 1
        if re.search(pxml, one_changed_file):
            xml_count = xml_count + 1
        if re.search(pconfig, one_changed_file):
            config_count = config_count + 1
    javaNum = java_count + rb_count
    configNum = xml_count + config_count
    return javaNum, configNum


# 将所有commit信息整合成只有构建的数值形式
def commitDataIntoNum(filePath9, filePath10):
    info_data = openJsonFile(filePath9)

    final_data = []
    p1 = re.compile('\d+')
    psuccess = re.compile('.*passed.*')
    last_build_result = "0"
    for info_data_item in info_data:
        if info_data_item["build_result"]:
            last_build_time = int(re.search(p1, info_data_item["commit_time"]).group())
            break
    mydic = {"additions_num": "0 additions",
             "deletions_num": "0 deletions",
             "changed_file_num": "0 changed files",
             "commit_title_length": "0",
             "commit_description_length": "0",
             "java_num": "0",
             "config_num": "0",
             "commit_count": "0",
             "last_build_result": "0",
             "time_interval": "0",
             }

    for info_data_item in info_data:
        mydic["additions_num"] = str(int(re.search(p1, mydic["additions_num"]).group()) + int(
            re.search(p1, info_data_item["additions_num"]).group())).decode('utf-8')
        mydic["deletions_num"] = str(int(re.search(p1, mydic["deletions_num"]).group()) + int(
            re.search(p1, info_data_item["deletions_num"]).group())).decode('utf-8')
        mydic["changed_file_num"] = str(int(re.search(p1, mydic["changed_file_num"]).group()) + int(
            re.search(p1, info_data_item["changed_file_num"]).group())).decode('utf-8')
        mydic["commit_title_length"] = str(
            int(re.search(p1, mydic["commit_title_length"]).group()) + len(info_data_item["commit_title"])).decode(
            'utf-8')
        mydic["commit_description_length"] = str(int(re.search(p1, mydic["commit_description_length"]).group()) + len(
            info_data_item["commit_description"])).decode('utf-8')
        javaNumCal, configNumCal = getJavaConfigNum(info_data_item)
        mydic["java_num"] = str(int(re.search(p1, mydic["java_num"]).group()) + javaNumCal).decode('utf-8')
        mydic["config_num"] = str(int(re.search(p1, mydic["config_num"]).group()) + configNumCal).decode('utf-8')
        mydic["commit_count"] = str(int(re.search(p1, mydic["commit_count"]).group()) + 1).decode('utf-8')

        if info_data_item["build_result"]:
            mydic["author_name"] = info_data_item["author_name"]
            mydic["commit_href"] = info_data_item["commit_href"]
            mydic["commit_id"] = info_data_item["commit_id"]
            mydic["commit_time"] = info_data_item["commit_time"]
            mydic["auther_commit_total"] = info_data_item["auther_commit_total"]
            mydic["commit_tag"] = info_data_item["commit_tag"]

            mydic["length_all_description"] = str(int(re.search(p1, mydic["commit_title_length"]).group()) + int(
                re.search(p1, mydic["commit_description_length"]).group())).decode('utf-8')
            mydic["changed_code_lines"] = str(int(re.search(p1, mydic["additions_num"]).group()) + int(
                re.search(p1, mydic["deletions_num"]).group())).decode('utf-8')
            mydic["average_commit_filenum"] = str(int(re.search(p1, mydic["changed_file_num"]).group()) // int(
                re.search(p1, mydic["commit_count"]).group())).decode('utf-8')

            mydic["last_build_result"] = last_build_result
            mydic["time_interval"] = str(
                int(re.search(p1, info_data_item["commit_time"]).group()) - last_build_time).decode('utf-8')

            if re.search(psuccess, info_data_item["build_result"]):
                mydic["build_result"] = "1"
                last_build_result = "1"
            else:
                mydic["build_result"] = "0"
                last_build_result = "0"
            last_build_time = int(re.search(p1, info_data_item["commit_time"]).group())

            final_data.append(mydic)
            mydic = {"additions_num": "0 additions", "deletions_num": "0 deletions",
                     "changed_file_num": "0 changed files", "commit_title_length": "0",
                     "commit_description_length": "0", "java_num": "0", "config_num": "0", "commit_count": "0",
                     "last_build_result": "0", "time_interval": "0"}

    with open(filePath10, 'w') as json_file:
        json_file.write(json.dumps(sorted(final_data, key=lambda x: x['commit_time']), indent=4))


# filePath1 testresult filepath2 commitintomerge
def build_relation_with_last_one(file_path1, file_path2, file_path3):
    info_data = openJsonFile(file_path1)

    final_data = []
    p1 = re.compile('\d+')
    for i in range(1, len(info_data)):
        dic = {}

        history_failure_time = 0
        history_success_time = 0
        for j in range(0, i-1):
            if info_data[j]["build_result"]== "0":
                history_failure_time+=1
            else:
                history_success_time+=1

        dic["history_failure_time"]=history_failure_time
        dic["history_success_time"]=history_success_time

        this_build = info_data[i]
        this_build_time = int(re.search(p1, this_build["commit_time"]).group())

        dic["commit_id"] = this_build["commit_id"]
        dic["commit_time"] = str(re.search(p1, this_build["commit_time"]))

        last_build = info_data[i - 1]
        last_build_time = int(re.search(p1, last_build["commit_time"]).group())

        file_list = {}
        for commit_info in openJsonFile(file_path2):
            commit_time = int(re.search(p1, commit_info["commit_time"]).group())
            if last_build_time < commit_time <= this_build_time:
                for file_name in commit_info["changed_file"]:
                    if file_name in file_list:
                        file_list[file_name].append(commit_info["author_name"])

                    else:
                        file_list[file_name] = []
                        file_list[file_name].append(commit_info["author_name"])
            else:
                pass

        people_list = []
        collision_file = 0
        for filenum in file_list.values():
            if len(filenum) > 1:
                collision_file += 1
                for people in filenum:
                    if people in people_list:
                        pass
                    else:
                        people_list.append(people)

        dic["collision_people"] = int(len(people_list))
        dic["collision_file"] = int(collision_file)

        failure_time_interval = 0
        j = i
        while j >= 0:
            if int(re.search(p1, info_data[j]["build_result"]).group()) == 0:
                failure_time_interval = this_build_time - int(re.search(p1, info_data[j]["commit_time"]).group())
                break
            else:
                j = j - 1
                continue
        dic["failure_time_interval"] = int(failure_time_interval)

        final_data.append(dic)
    with open(file_path3, 'w') as json_file:
        json_file.write(json.dumps(final_data))


# 最近五次项目构建的成功率
def build_success_rate_five(filePath11, filePath12):
    info_data = openJsonFile(filePath11)

    final_data = []
    p1 = re.compile('\d+')
    mydic = {}
    build_tag = 0

    for info_data_item in info_data:
        mydic["commit_time"] = info_data_item["commit_time"]
        if build_tag >= 5:
            success = 0
            for i in range(build_tag - 5, build_tag):
                if int(re.search(p1, info_data[i]["build_result"]).group()) == 1:
                    success = success + 1
            success_rate = success / 5.0
            mydic["success_last_five"] = str(success_rate).decode("utf-8")
        else:
            mydic["success_last_five"] = "0"
        mydic["build_tag"] = str(build_tag).decode("utf-8")
        build_tag = build_tag + 1

        final_data.append(mydic)
        mydic = {}

    with open(filePath12, 'w') as json_file:
        json_file.write(json.dumps(sorted(final_data, key=lambda x: x['commit_time']), indent=4))


# 将项目最近五次的构建成功率与构建信息合并
def merge_info_success_rate(filePath13, filePath14, filePath15):
    project_success_data = openJsonFile(filePath13)
    info_data = openJsonFile(filePath14)
    commit_info_merge_success = []

    for project_success_data_item in project_success_data:
        for info_data_item in info_data:
            if str(info_data_item["commit_time"]) == str(project_success_data_item["commit_time"]):
                info_data_item["build_tag"] = project_success_data_item["build_tag"]
                info_data_item["success_last_five"] = project_success_data_item["success_last_five"]
                commit_info_merge_success.append(info_data_item)
                break

        with open(filePath15, 'w') as json_file:
            json_file.write(json.dumps(sorted(commit_info_merge_success, key=lambda x: x['commit_time']), indent=4))


def predictData(filePath16, filePath17, filePath18):
    commit_info_data = openJsonFile(filePath16)
    build_info_data = openJsonFile(filePath17)

    p1 = re.compile('\d+')
    mydic = {"additions_num": "0 additions", "deletions_num": "0 deletions", "changed_file_num": "0 changed files",
             "commit_title_length": "0", "commit_description_length": "0", "java_num": "0", "config_num": "0",
             "commit_count": "0", "last_build_result": "0", "time_interval": "0"}
    print str(build_info_data[-1]["commit_id"])
    for commit_file_info_item in commit_info_data:
        if str(commit_file_info_item["commit_id"]) == str(build_info_data[-1]["commit_id"]):
            for i in range(int(re.search(p1, commit_file_info_item["commit_tag"]).group()),
                           int(re.search(p1, commit_info_data[-2]["commit_tag"]).group())):
                mydic["additions_num"] = str(int(re.search(p1, mydic["additions_num"]).group()) + int(
                    re.search(p1, commit_info_data[i]["additions_num"]).group())).decode('utf-8')
                mydic["deletions_num"] = str(int(re.search(p1, mydic["deletions_num"]).group()) + int(
                    re.search(p1, commit_info_data[i]["deletions_num"]).group())).decode('utf-8')
                mydic["changed_file_num"] = str(int(re.search(p1, mydic["changed_file_num"]).group()) + int(
                    re.search(p1, commit_info_data[i]["changed_file_num"]).group())).decode('utf-8')
                mydic["commit_title_length"] = str(int(re.search(p1, mydic["commit_title_length"]).group()) + len(
                    str(commit_info_data[i]["commit_title"]))).decode('utf-8')
                mydic["commit_description_length"] = str(
                    int(re.search(p1, mydic["commit_description_length"]).group()) + len(
                        str(commit_info_data[i]["commit_description"]))).decode('utf-8')
                javaNumCal, configNumCal = getJavaConfigNum(commit_info_data[i])
                mydic["java_num"] = str(int(re.search(p1, mydic["java_num"]).group()) + javaNumCal).decode('utf-8')
                mydic["config_num"] = str(int(re.search(p1, mydic["config_num"]).group()) + configNumCal).decode(
                    'utf-8')
                mydic["commit_count"] = str(int(re.search(p1, mydic["commit_count"]).group()) + 1).decode('utf-8')
            break

    if (int(re.search(p1, mydic["commit_count"]).group()) != 0):
        mydic["author_name"] = commit_info_data[-1]["author_name"]
        mydic["commit_href"] = commit_info_data[-1]["commit_href"]
        mydic["commit_id"] = commit_info_data[-1]["commit_id"]
        mydic["commit_time"] = commit_info_data[-1]["commit_time"]
        mydic["auther_commit_total"] = commit_info_data[-1]["auther_commit_total"]
        mydic["commit_tag"] = commit_info_data[-1]["commit_tag"]

        mydic["length_all_description"] = str(int(re.search(p1, mydic["commit_title_length"]).group()) + int(
            re.search(p1, mydic["commit_description_length"]).group())).decode('utf-8')
        mydic["changed_code_lines"] = str(int(re.search(p1, mydic["additions_num"]).group()) + int(
            re.search(p1, mydic["deletions_num"]).group())).decode('utf-8')
        mydic["average_commit_filenum"] = str(int(re.search(p1, mydic["changed_file_num"]).group()) // int(
            re.search(p1, mydic["commit_count"]).group())).decode('utf-8')
        mydic["last_build_result"] = build_info_data[-1]["build_result"]
        mydic["time_interval"] = str(int(re.search(p1, commit_info_data[-1]["commit_time"]).group()) - int(
            re.search(p1, build_info_data[-1]["commit_time"]).group())).decode('utf-8')
        mydic["build_result"] = str("0").decode('utf-8')
        mydic["success_last_five"] = build_info_data[-1]["success_last_five"]

        build_info_data.append(mydic)

    with open(filePath18, 'w') as json_file:
        json_file.write(json.dumps(sorted(build_info_data, key=lambda x: x['commit_time']), indent=4))

# file_path1 test_result
# def normal_distribution(file_path1):
#     build_info_data = openJsonFile(file_path1)
#
#     FtOfFailure = []
#     for i in range(0, len(build_info_data)):
#         if build_info_data[i]["build_result"]=="0":
#             distance=int(len(build_info_data)) - i
#             FtOfFailure.append(distance)
#         else:
#             pass
#     means = np.mean(file_path1)
    


def dataProcess(filePath1, filePath2):
    filePath3 = '../data/commit_info_merge.json'
    filePath4 = '../data/contributer_info.json'
    filePath5 = '../data/commit_info_merge_contributer.json'
    filePath6 = '../data/test_result.json'
    filePath7 = '../data/result_project_success.json'
    filePath8 = '../data/commit_info_merge_success.json'
    filePath9 = '../data/final.json'
    filePath10 = '../data/relation_with_last_one.json'
    # 将构建结果和链接  与  commit各项信息合并
    commits_info_merge(filePath1, filePath2, filePath3)
    print "success"

    # 获取每个contributer提交的总次数，制成新文件
    get_contributer_info(filePath3, filePath4)
    print "success"

    # 把contributer提交的总次数与commit的信息合并
    merge_contributer_commit(filePath4, filePath3, filePath5)
    print "success"

    # 将所有commit信息整合成只有构建的数值形式
    commitDataIntoNum(filePath5, filePath6)
    print "success"

    # filePath1 testresult filepath2 commitintomerge
    build_relation_with_last_one(filePath6, filePath3, filePath10)
    print "success"

    # 最近五次项目构建的成功率
    build_success_rate_five(filePath6, filePath7)
    print "success"

    # 将项目最近五次的构建成功率与构建信息合并
    merge_info_success_rate(filePath7, filePath6, filePath8)
    print "success"

    # 将最后几条没有构建的组成最后一条预测信息
    predictData(filePath5, filePath8, filePath9)
    print "success"


if __name__ == '__main__':
    # reload(sys)
    # sys.setdefaultencoding('utf-8')
    # filePath1 = sys.argv[1]
    # filePath2 = sys.argv[2]
    # print filePath1
    # print type(filePath1)
    # # filePath1 = "../data/commit_info.json"
    # # filePath2 = "../data/project_commit.json"
    # # print filePath2
    # # print type(filePath2)
    # dataProcess(filePath1,filePath2)
    # print "success"
    filePath1 = "../data/commit_info.json"
    filePath2 = "../data/project_commit.json"
    dataProcess(filePath1, filePath2)
