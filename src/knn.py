import numpy as np

class KNN(object):
    def __init__(self,data,k):
        self.k = k
        self.data = np.array(data)

    def build(self):
        pass

    def dis(self,a,b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def predict(self,x):
        result = []
        for item in self.data:
            result.append([self.dis(item[:2],x),item[2]])
        result = sorted(result)
        count = {}
        for i in range(0,self.k):
            vote = result[i][1]
            count[vote] = count.get(vote,0) + 1
        ans = [0,0]
        for label in count:
            if count[label] > ans[0]:
                ans = [count[label],label]
        return ans[1]
