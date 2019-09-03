import os
import random

imgs = os.listdir("Images")

random.shuffle(imgs)

train_list = imgs[:2000]
test_list = imgs[2000:]

train_list.sort()
test_list.sort()

with open("train_list.txt",'w') as fw:
    for item in train_list:
        path = "Images/" + item
        line = path + "\n"
        fw.write(line)

with open("test_list.txt",'w') as fw:
    for item in test_list:
        path = "Images/" + item
        line = path + "\n"
        fw.write(line)
        
