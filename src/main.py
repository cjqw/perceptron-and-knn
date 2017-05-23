#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from perceptron import Perceptron
from knn import KNN

TEST_SET_FILE = "hw5test.csv"
TRAIN_SET_FILE = "hw5train.csv"

def parseItem(item):
    return list(map(float,item.split(',')))

def readCSV(fName):
    with open("data/" + fName,"r") as fin:
        st = fin.readlines()[1:]
    return list(map(parseItem,st))

def drawGraph(data,predict,name = ""):
    hit = 0
    for p,l in zip(predict,data):
        if p * l[2] > 0 : hit += 1
    accuracy = hit / len(data)
    print(predict)
    print("ACCURACY",accuracy)

def testPerceptron():
    global train_set,test_set
    model = Perceptron(train_set)
    model.train(0.01)
    predict = [model.predict(item) for item in test_set]
    drawGraph(test_set,predict,"Perception")

def testKNN():
    global train_set,test_set
    model = KNN(train_set,3)
    model.build()
    predict = [model.predict(item) for item in test_set]
    drawGraph(test_set,predict,"KNN")

train_set = readCSV(TRAIN_SET_FILE)
test_set = readCSV(TEST_SET_FILE)

testPerceptron()
testKNN()
