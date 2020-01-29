import sys
import os
import numpy as np
import argparse
import re
from pathlib import Path
import random
import yaml

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("textfile1", help="text file1 you want to marge")
    parser.add_argument("textfile2", help="text file2 you want to marge")
    parser.add_argument("margefile", help="output")
    args = parser.parse_args()
    return args

args = ParseArgs()

def readlines_file(file_name):
    # 行毎のリストを返す
    with open(file_name, 'r') as file:
        return file.readlines()


def save_file(file_name, text):
    with open(file_name, 'a') as file:
        file.write(text + "\n")

# 読み込んだファイルをlist型で受け取る
cal1 = readlines_file(args.textfile1)
cal2 = readlines_file(args.textfile2)

# 改行や空白文字を削除
cal1 = list(map(lambda x: x.strip(), cal1))
cal2 = list(map(lambda x: x.strip(), cal2))

# タブ区切りで並べたリストを作成
lines = ["{0}\t{1}".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

for line in lines:
    save_file(args.margefile,line)

