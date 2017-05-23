import numpy as np

class Perceptron(object):
    def __init__(self,data):
        length = len(data)
        self.data = np.array(data)
        self.w = np.array([1,1])
        self.b = 0

    def train_batch(self,rate):
        for item in self.data:
            z = self.predict(item)
            label = item[2]
            if label * z < 0:
                self.w = self.w + rate * item[:2] * label
                self.b = self.b + label

    def train(self,rate):
        for i in range(0,100):
            self.train_batch(rate)

    def predict(self,item):
        z = sum(self.w * item[:2]) + self.b
        if z > 0:
            return 1
        else:
            return -1
