# encoding=utf8
import sys
import json
import re
import datetime
import time
import pytz
from collections import deque
import math
import os
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

def get_each_kind_num(info_data_item):
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


# 数据处理
def data_processing(filePath1, filePath2, filePath3):
    project_commit_data = openJsonFile(filePath1)
    commit_info_data = openJsonFile(filePath2)

    p1 = re.compile('[^/]+(?!.*/)')

    final_info_merge = []

    help = 1

    # processing 1: basic information
    for project_commit_item in project_commit_data:

        # ad = 1
        print help
        help += 1

        commit_id = re.search(p1, project_commit_item["commit_href"]).group()
        for commit_info_item in commit_info_data:

            final_dict = {}

            if commit_info_item["commit_id"]:
                this_commit_id = commit_info_item["commit_id"][0]
            if commit_id == this_commit_id:
                final_dict["commit_href"] = project_commit_item["commit_href"]

                if project_commit_item["build_result"] == "Commit: passed":
                    final_dict["build_result"] = 1
                elif project_commit_item["build_result"] == "Commit: failed":
                    final_dict["build_result"] = 0
                else:
                    final_dict["build_result"] = project_commit_item["build_result"]

                final_dict["commit_id"] = this_commit_id
                final_dict["additions_num"] = int((commit_info_item["additions_num"][0].split(' '))[0])
                # final_dict["commit_title"] = commit_info_item["commit_title"][0]
                # final_dict["commit_title_length"] = len(commit_info_item["commit_title"][0])
                final_dict["author_name"] = commit_info_item["author_name"][0]
                final_dict["changed_file"] = commit_info_item["changed_file"]
                if int((commit_info_item["changed_file_num"][0].split(' '))[0]) != 0:
                    final_dict["changed_file_num"] = math.log(int((commit_info_item["changed_file_num"][0].split(' '))[0]))
                else:
                    final_dict["changed_file_num"] = 0
                # final_dict["changed_file_num"] = int((commit_info_item["changed_file_num"][0].split(' '))[0])
                final_dict["deletions_num"] = int((commit_info_item["deletions_num"][0].split(' '))[0])
                if len(commit_info_item["commit_description"]) > 0:
                    final_dict["commit_msg_length"] = math.log(len(commit_info_item["commit_title"][0]) + len(commit_info_item["commit_description"][0]))
                else:
                    final_dict["commit_msg_length"] = math.log(len(commit_info_item["commit_title"][0]))
                # if len(final_dict["commit_description"])>0:
                #     final_dict["commit_description_length"] = math.log(len(final_dict["commit_description"]))

                info_time = commit_info_item["commit_time"][0]
                local_tz = pytz.timezone('Asia/Chongqing')
                utc_dt = datetime.datetime.strptime(info_time, '%Y-%m-%dT%H:%M:%SZ')
                local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
                final_dict["hour_time"] = local_dt.hour
                final_dict["month_day"] = local_dt.day
                final_dict["weekday"] = local_dt.weekday()   # 0-6 from Mon-Sun
                final_dict["commit_time"] = utc_to_local(info_time)

                final_info_merge.append(final_dict)
                break

    final_info_merge = sorted(final_info_merge, key=lambda x: x['commit_time'], reverse=False)

    # for commit_info in final_info_merge:
    #      print commit_info["commit_time"]
    # for i in range(0, len(final_info_merge)):
    #     print final_info_merge[i]["commit_time"]

    print "process 1 finish"

    help = 1

    # processing 2: build info with last one
    last_build_time = 0
    last_committer = ""
    file_list_after_last_build = {}
    build_info = {
        "last_build_result": "first build",
        "time_interval_after_last_build": 0,
        "additions_num_after_last_build": 0,
        "deletions_num_after_last_build": 0,
        "changed_file_num_after_last_build": 0,
        "changed_java_file_num_after_last_build": 0,
        "changed_config_file_num_after_last_build": 0,
        "src_config_file_num": 0,
        "commit_num_after_last_build": 0,
        "average_filenum_per_commit_after_last_build": 0,
        "same_committer": 0,
        "collision_people_num": 0,
        "collision_file_num": 0
    }
    for commit_info in final_info_merge:

        print help
        help += 1

        build_info["additions_num_after_last_build"] += commit_info["additions_num"]
        build_info["deletions_num_after_last_build"] += commit_info["deletions_num"]
        build_info["changed_file_num_after_last_build"] += commit_info["changed_file_num"]

        java_num_cal, config_num_cal = get_each_kind_num(commit_info)

        for file_name in commit_info["changed_file"]:
            if file_name in file_list_after_last_build:
                file_list_after_last_build[file_name].append(commit_info["author_name"])
            else:
                file_list_after_last_build[file_name] = []
                file_list_after_last_build[file_name].append(commit_info["author_name"])

        build_info["changed_java_file_num_after_last_build"] += java_num_cal
        build_info["changed_config_file_num_after_last_build"] += config_num_cal
        build_info["commit_num_after_last_build"] += 1

        if commit_info["build_result"] == 1 or commit_info["build_result"] == 0:

            if build_info["changed_java_file_num_after_last_build"] > 0:
                build_info["changed_java_file_num_after_last_build"] = 1
            else:
                build_info["changed_java_file_num_after_last_build"] = 0

            if build_info["changed_config_file_num_after_last_build"] > 0:
                build_info["changed_config_file_num_after_last_build"] = 1
            else:
                build_info["changed_config_file_num_after_last_build"] = 0

            if build_info["changed_java_file_num_after_last_build"]+build_info["changed_config_file_num_after_last_build"] > 0:
                build_info["src_config_file_num"] = 1
            else:
                build_info["src_config_file_num"] = 0

            if build_info["changed_file_num_after_last_build"] != 0:
                build_info["average_filenum_per_commit_after_last_build"] = math.log(round(build_info["changed_file_num_after_last_build"] / build_info["commit_num_after_last_build"], 2))
            else:
                build_info["average_filenum_per_commit_after_last_build"] = 0

            if build_info["additions_num_after_last_build"] != 0:
                build_info["additions_num_after_last_build"] = math.log(build_info["additions_num_after_last_build"])
            else:
                build_info["additions_num_after_last_build"] = 0

            if build_info["deletions_num_after_last_build"] != 0:
                build_info["deletions_num_after_last_build"] = math.log(build_info["deletions_num_after_last_build"])
            else:
                build_info["deletions_num_after_last_build"] = 0

            if build_info["commit_num_after_last_build"] != 0:
                build_info["commit_num_after_last_build"] = math.log(build_info["commit_num_after_last_build"])
            else:
                build_info["commit_num_after_last_build"] = 0

            if last_build_time == 0:
                build_info["time_interval_after_last_build"] = 0
            else:
                if commit_info["commit_time"] - last_build_time != 0:
                    # print last_build_time
                    # print commit_info["commit_time"]
                    # print "---------"
                    build_info["time_interval_after_last_build"] = math.log(abs(commit_info["commit_time"]-last_build_time))-math.log(86400)
                    # print build_info["time_interval_after_last_build"]
                    # print "........."
                else:
                    build_info["time_interval_after_last_build"] = 0

            if last_committer == commit_info["author_name"]:
                build_info["same_committer"] = 1
            else:
                build_info["same_committer"] = 0

            people_list = []
            collision_file = 0
            for filenum in file_list_after_last_build.values():
                if len(filenum) > 1:
                    collision_file += 1
                    for people in filenum:
                        if people in people_list:
                            pass
                        else:
                            people_list.append(people)

            if int(len(people_list))>0:
                build_info["collision_people_num"] = math.log(int(len(people_list)))
            else:
                build_info["collision_people_num"] = 0

            if collision_file>0:
                build_info["collision_file_num"] = math.log(int(collision_file))
            else:
                build_info["collision_file_num"] = 0

            # commit_info["build_info_relation_with_last_one"] = build_info
            commit_info["last_build_result"] = build_info["last_build_result"]
            commit_info["time_interval_after_last_build"] = build_info["time_interval_after_last_build"]
            commit_info["additions_num_after_last_build"] = build_info["additions_num_after_last_build"]
            commit_info["deletions_num_after_last_build"] = build_info["deletions_num_after_last_build"]
            # commit_info["changed_file_num_after_last_build"] = build_info["changed_file_num_after_last_build"]
            commit_info["changed_java_file_num_after_last_build"] = build_info["changed_java_file_num_after_last_build"]
            commit_info["changed_config_file_num_after_last_build"] = build_info["changed_config_file_num_after_last_build"]
            commit_info["src_config_file_num"] = build_info["src_config_file_num"]
            commit_info["commit_num_after_last_build"] = build_info["commit_num_after_last_build"]
            commit_info["average_filenum_per_commit_after_last_build"] = build_info["average_filenum_per_commit_after_last_build"]
            commit_info["same_committer"] = build_info["same_committer"]
            commit_info["collision_people_num"] = build_info["collision_people_num"]
            commit_info["collision_file_num"] = build_info["collision_file_num"]

            build_info = {
                "last_build_result": commit_info["build_result"],
                "time_interval_after_last_build": 0,
                "additions_num_after_last_build": 0,
                "deletions_num_after_last_build": 0,
                "changed_file_num_after_last_build": 0,
                "changed_java_file_num_after_last_build": 0,
                "changed_config_file_num_after_last_build": 0,
                "src_config_file_num": 0,
                "commit_num_after_last_build": 0,
                "average_filenum_per_commit_after_last_build": 0,
                "same_committer": 0,
                "collision_people_num": 0,
                "collision_file_num": 0
            }
            file_list_after_last_build = {}
            last_committer = commit_info["author_name"]
            last_build_time = commit_info["commit_time"]

    print "process 2 finish"

    help = 1

    # processing 3: build info with history
    history_builds_info = {
        "history_success_time": 0,
        "history_failure_time": 0,
        "time_interval_after_last_failure": 0
    }
    for commit_info in final_info_merge:

        print help
        help += 1

        if commit_info["build_result"] != 1 and commit_info["build_result"] != 0:
            continue

        i = final_info_merge.index(commit_info)
        if i == 0:
            commit_info["history_builds_info"] = history_builds_info
            continue
        for j in range(0, i-1):
            if final_info_merge[j]["build_result"] == 1:
                history_builds_info["history_success_time"] += 1
            elif final_info_merge[j]["build_result"] == 0:
                history_builds_info["history_failure_time"] += 1
            else:
                pass

        failure_time_interval = 0
        j = i - 1
        while j >= 0:
            if final_info_merge[j]["build_result"] == 0:
                # print commit_info["commit_time"]
                # print final_info_merge[j]["commit_time"]
                # print ".........."
                failure_time_interval = abs(commit_info["commit_time"] - final_info_merge[j]["commit_time"])
                break
            else:
                j = j - 1
                continue

        # print failure_time_interval
        # print "-----------"
        if failure_time_interval != 0:
            history_builds_info["time_interval_after_last_failure"] = math.log(failure_time_interval)-math.log(86400)
            # print history_builds_info["time_interval_after_last_failure"]
            # print "////////"
        else:
            history_builds_info["time_interval_after_last_failure"] = 0

        # commit_info["history_builds_info"] = history_builds_info
        if history_builds_info["history_success_time"]+history_builds_info["history_failure_time"]==0:
            commit_info["project_history"] = 0
        else:
            commit_info["project_history"] = (history_builds_info["history_failure_time"]+0.0)/(history_builds_info["history_success_time"] + history_builds_info["history_failure_time"])
        commit_info["time_interval_after_last_failure"] = history_builds_info["time_interval_after_last_failure"]

        history_builds_info = {
            "history_success_time": 0,
            "history_failure_time": 0,
            "time_interval_after_last_failure": 0
        }

    print "process 3 finish"

    help = 1

    # process 4: author info
    author_info = {
        "total_commit_before": 0,
        "total_build_before": 0,
        "build_success_num": 0,
        "build_fail_num": 0,
        "committer_failure_rate_recent": 0
    }
    developer_list = []
    for commit_info in final_info_merge:

        print help
        help += 1

        if commit_info["build_result"] != 1 and commit_info["build_result"] != 0:
            continue

        author = commit_info["author_name"]
        developer_list.append(author)
        i = final_info_merge.index(commit_info)
        if i == 0:
            commit_info["developer_count"] = 1
            commit_info["author_info"] = author_info
            continue

        success = 0.0
        failure = 0.0
        for j in range(i-1, -1, -1):
            if final_info_merge[j]["author_name"] == author:
                author_info["total_commit_before"] += 1
                if final_info_merge[j]["build_result"] == 1:
                    author_info["total_build_before"] += 1
                    author_info["build_success_num"] += 1

                    if success + failure >= 5:
                        continue
                    else:
                        success += 1

                else:
                    author_info["total_build_before"] += 1
                    author_info["build_fail_num"] += 1

                    if success + failure >= 5:
                        continue
                    else:
                        failure += 1
            else:
                pass

            new_author = final_info_merge[j]["author_name"]
            if new_author in developer_list:
                pass
            else:
                developer_list.append(new_author)

        if success+failure == 0:
            author_info["committer_failure_rate_recent"] = 0
        else:
            author_info["committer_failure_rate_recent"] = failure/(failure+success)

        if len(developer_list)>0:
            commit_info["developer_count"] = math.log(len(developer_list))
        else:
            commit_info["developer_count"] = 0

        # commit_info["author_info"] = author_info
        if author_info["total_commit_before"] > 0:
            commit_info["total_commit_before"] = math.log(author_info["total_commit_before"])
        else:
            commit_info["total_commit_before"] = 0

        # commit_info["total_build_before"] = author_info["total_build_before"]
        if author_info["build_success_num"] + author_info["build_fail_num"] == 0:
            commit_info["committer_history"] = 0
        else:
            commit_info["committer_history"] = (author_info["build_fail_num"]+0.0)/(author_info["build_success_num"] + author_info["build_fail_num"]+0.0)
        commit_info["committer_failure_rate_recent"] = author_info["committer_failure_rate_recent"]

        author_info = {
            "total_commit_before": 0,
            "total_build_before": 0,
            "build_success_num": 0,
            "build_fail_num": 0,
            "committer_failure_rate_recent": 0
        }
        developer_list = []

    print "process 4 finish"

    help = 1

    # process 5: normal distribution
    for commit_info in final_info_merge:

        print help
        help += 1

        if commit_info["build_result"] != 1 and commit_info["build_result"] != 0:
            continue

        i = final_info_merge.index(commit_info)
        if i == 0:
            continue

        build_index = []
        build_result = []
        for j in range(0, i-1):
            if final_info_merge[j]["build_result"] != 1 and final_info_merge[j]["build_result"] != 0:
                continue
            else:
                build_index.append(j)
                build_result.append(final_info_merge[j]["build_result"])

        result = 0.0
        for x in range(0, len(build_index)):
            if build_result[x] == 0:
                ft = len(build_result)-x  # contain this build
                Dt = 0.0
                for y in range(x+1, len(build_index)):
                    Dt = Dt+(build_index[y]-build_index[x])
                dt = Dt/ft
                if ft == 0 or dt == 0:
                    result += 0
                else:
                    result += (1/(math.sqrt(2.0*math.pi)*dt)) * math.exp(0.0-math.pow(ft, 2)/(math.pow(dt, 2)*2.0))
            else:
                pass
        commit_info["gaussian_threat_from_histroy"] = result

    print "process 5 finish"

    help = 1

    # process 6: failure_last_five
    for commit_info in final_info_merge:

        print help
        help += 1

        if commit_info["build_result"] != 1 and commit_info["build_result"] != 0:
            continue

        i = final_info_merge.index(commit_info)
        if i == 0:
            continue

        success = 0.0
        failure = 0.0
        for j in range(i-1, -1, -1):
            if success+failure >= 5:
                break
            if final_info_merge[j]["build_result"] == 1:
                success += 1
            elif final_info_merge[j]["build_result"] == 0:
                failure += 1
            else:
                pass

        if success+failure == 0.0:
            commit_info["failure_rate_last_five"] = 0
        else:
            commit_info["failure_rate_last_five"] = failure/(success+failure)

    print "process 6 finish"

    final_build_info=[]
    for commit_info in final_info_merge:

        if commit_info["build_result"] == 1 or commit_info["build_result"] == 0:
            if commit_info["last_build_result"] == "first build":
                pass
            else:
                commit_info.pop("additions_num")
                commit_info.pop("deletions_num")
                commit_info.pop("changed_file")
                commit_info.pop("commit_id")
                final_build_info.append(commit_info)
        else:
            continue

    with open(filePath3, 'w') as input_file:
        input_file.write(json.dumps(sorted(final_build_info, key=lambda x: x['commit_time']), indent=4))


def dataProcess(project_name):
    project_commit_file = '../SpiderData/'+project_name+'.json'
    commits_info_file = '../SpiderData/'+project_name+'_commitInfo.json'
    final_file_name = '../data/'+project_name+'_final.json'

    data_processing(project_commit_file, commits_info_file, final_file_name)


if __name__ == '__main__':

    files = os.listdir("../SpiderData")

    arr = []
    for json_file in files:
        if json_file.split(".")[1] == "json" and ((json_file.split(".")[0]).split("_"))[-1] != "commitInfo":
            arr.append(json_file.split(".")[0])

    for project_name in arr:
        print project_name
        dataProcess(project_name)

    # dataProcess("gitlab-org_gitaly")

    # print math.log(86400)
