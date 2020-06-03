import re
import urllib.request
import requests
import urllib.parse
import os
import threading
from tkinter import *

class Bing():

    def __init__(self):

        self.picture_urls = []

    def get_urls(self,key):

        count = 1
        key = urllib.parse.quote(key)
        while count<500:
            root_url = 'https://www.bing.com/images/async?q=%s&async=content&first=%d'%(key,count)
            html = requests.get(root_url).text
            self.picture_urls.extend(re.findall('murl&quot;:&quot;(http://.*?\.jpg)', html))
            count += 36
        return self.picture_urls

    def down(self,path,num,urls,screen):

        path += '/search_picture/'
        if urls is None or num == 0:
            return
        try:
            os.makedirs(path)
        except:
            print(INSERT,"文件夹创建失败")
            return

        count = 1

        for url in urls:
            save_path = path+"%d.jpg"%count
            try:
                urllib.request.urlretrieve(url,save_path)
                print(INSERT,"正在下载第%02d张：%s\n"%(count,url))
                count += 1
                if count>num:
                    break
            except:
                print(INSERT,"当前图片链接不可用\n")
                continue
        print(INSERT,"下载完毕\n")
        return

    def run(self,path,num,key,screen):

        pic_urls = self.get_urls(key)
        p = threading.Thread(target=self.down,args=(path,num,pic_urls,screen))
        p.start()




if __name__ == '__main__':
    root = "./downloads-bing"
    keys = ["tiger"]
    for key in keys:
        print("begin key: ", key)
        master = Bing()
        user_path = os.path.join(root, key)
        user_num = 500
        user_key = key
        screen = None
        master.run(user_path, user_num, user_key, screen)