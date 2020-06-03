import numpy as np
import os



# dir = r"D:\Project\Project2019\DataBias2019\Project\data\Dog-Cat\result\logistic"
dir = r"D:\Project\Project2019\DataBias2019\Project\data\Dog-Cat\result\svm"

filename_list = os.listdir(dir)

for filename in filename_list:
    # print(filename)
    file = open(os.path.join(dir, filename), "r").read()
    file = file.strip("\n").split("\n")
    for line in file:
        try:
            a = float(line.split(",")[2])
            if a > 0.64:
                print(line.replace(",", "\t"))
        except:
            None