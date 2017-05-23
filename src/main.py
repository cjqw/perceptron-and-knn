#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from util.util import *
from perceptron import Perceptron
from knn import KNN
import matplotlib.pyplot as plt
import numpy as np

TEST_SET_FILE = "hw5test.csv"
TRAIN_SET_FILE = "hw5train.csv"

def parseItem(item):
    return mapv(float,item.split(','))

def readCSV(fName):
    with open("data/" + fName,"r") as fin:
        st = fin.readlines()[1:]
    return mapv(parseItem,st)

def drawGraph(data,f,name = ""):
    # Draw contour
    delta = 0.02
    x = np.arange(3, 7, delta)
    y = np.arange(-2, 8, delta)
    X,Y = np.meshgrid(x, y)
    xx,yy = X.ravel(), Y.ravel()
    Z = np.array([f([x,y]) for x,y in zip(xx,yy)]).reshape(X.shape)
    CS = plt.contour(X, Y, Z,cmap=plt.cm.Paired
                     ,levels = [0])
    plt.clabel(CS)

    # Draw data
    data = partition(data,getValue(2))
    pos, neg = data[1], data[-1]
    plt.scatter(mapv(getValue(0),pos),mapv(getValue(1),pos))
    plt.scatter(mapv(getValue(0),neg),mapv(getValue(1),neg))
    plt.title(name + " Algorithm")
    print(name)
    plt.show()


def testPerceptron():
    global train_set,test_set
    model = Perceptron(train_set)
    model.train(0.01)
    drawGraph(test_set,model.predict,"Perceptron")

def testKNN():
    global train_set,test_set
    model = KNN(train_set,3)
    model.build()
    drawGraph(test_set,model.predict,"KNN")

train_set = readCSV(TRAIN_SET_FILE)
test_set = readCSV(TEST_SET_FILE)

testPerceptron()
testKNN()
