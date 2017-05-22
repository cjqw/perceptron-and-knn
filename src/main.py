#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from perception import Perception
from knn import KNN

TEST_SET_FILE = "hw5test.csv"
TRAIN_SET_FILE = "hw5train.csv"

def parseItem(item):
    return list(map(float,item.split(',')))

def readCSV(fName):
    with open("data/" + fName,"r") as fin:
        st = fin.readlines()[1:]
    return list(map(parseItem,st))

train_set = readCSV(TRAIN_SET_FILE)
test_set = readCSV(TEST_SET_FILE)
