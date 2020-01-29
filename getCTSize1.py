import sys
import os
import tensorflow as tf
from itertools import product
import numpy as np
import argparse
import yaml
import SimpleITK as sitk
from pathlib import Path

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", help="Input image file")
    parser.add_argument("listpath", help="The cut list")
    args = parser.parse_args()
    return args

args = ParseArgs()




num_totalpatches = os.path.getsize(args.imagefile)

print(num_totalpatches)
if(num_totalpatches < 150000000):
    fo = open(args.listpath, "a")
    fo.writelines(args.imagefile+"\n")
    fo.close()
