#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os,shutil
import re

from selenium import webdriver
import time
import urllib

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir


#输出目录
OUTPUT_DIR = os.path.join(config.raw_data_root, config.dog_cat, "dog-cage")
check_dir(OUTPUT_DIR)
#关键字数组：将在输出目录内创建以以下关键字们命名的txt文件
SEARCH_KEY_WORDS = ["dog cage"]
#页数
PAGE_NUM = 15

repeateNum = 0
preLen = 0

def getSearchUrl(keyWord):
    if(isEn(keyWord)):
        return 'https://www.google.com.hk/search?q=' + keyWord + '&safe=strict&source=lnms&tbm=isch'
    else:
        return 'https://www.google.com.hk/search?q=' + keyWord + '&safe=strict&hl=zh-CN&source=lnms&tbm=isch'

def isEn(keyWord):
    return all(ord(c) < 128 for c in keyWord)

# 启动Firefox浏览器
# driver = webdriver.Firefox(executable_path="C:/Users/Changjian/Anaconda3/Scripts")
driver = webdriver.Firefox()

if os.path.exists(OUTPUT_DIR) == False:
    os.makedirs(OUTPUT_DIR)

def output(SEARCH_KEY_WORD):
    global repeateNum
    global preLen

    print('搜索' + SEARCH_KEY_WORD + '图片中，请稍后...')

    # 如果此处为搜搜，搜索郁金香，此处可配置为：http://pic.sogou.com/pics?query=%D3%F4%BD%F0%CF%E3&di=2&_asf=pic.sogou.com&w=05009900&sut=9420&sst0=1523883106480
    # 爬取页面地址，该处为google图片搜索url
    url = getSearchUrl(SEARCH_KEY_WORD)

    # 如果是搜搜，此处配置为：'//div[@id="imgid"]/ul/li/a/img'
    # 目标元素的xpath，该处为google图片搜索结果内img标签所在路径
    xpath = '//div[@id="rg"]/div/div/a/img'

    # 浏览器打开爬取页面
    driver.get(url)

    outputFile = OUTPUT_DIR + '/' + SEARCH_KEY_WORD + '.txt'
    outputSet = set()

    # 模拟滚动窗口以浏览下载更多图片
    pos = 0
    m = 0 # 图片编号
    for i in range(PAGE_NUM):
        pos += i*600 # 每次下滚600
        js = "document.documentElement.scrollTop=%d" % pos
        driver.execute_script(js)
        time.sleep(1)
        for element in driver.find_elements_by_xpath(xpath):
            img_url = element.get_attribute('src')
            if img_url is not None and img_url.startswith('http'):
                outputSet.add(img_url)
        if preLen == len(outputSet):
            if repeateNum == 2:
                repeateNum = 0
                preLen = 0
                break
            else:
                repeateNum = repeateNum + 1
        else:
            repeateNum = 0
            preLen = len(outputSet)

    print('写入' + SEARCH_KEY_WORD + '图片中，请稍后...')
    file = open(outputFile, 'w')
    for val in outputSet:
        file.write(val + '\n')
    file.close()

    print(SEARCH_KEY_WORD+'图片搜索写入完毕')
    print(len(outputSet))

for val in SEARCH_KEY_WORDS:
    output(val)

driver.close()
