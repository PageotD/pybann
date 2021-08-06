import numpy as np
from pybann import Network
#import network
# Initialize network
size = (4, 8, 3) #(4, 8, 6, 3)
network = Network(size)


# Read data
with open('data/iris/iris.data', 'r') as f:
    lines = f.readlines()
    inData = []
    for line in lines:
        line = line.strip().split(",")
        if len(line) > 1:
            inValues = (float(line[0]), float(line[1]), float(line[2]), float(line[3]))
            if line[-1] == "Iris-setosa":
                attempted = (1, 0, 0)
            if line[-1] == "Iris-versicolor":
                attempted = (0, 1, 0)
            if line[-1] == "Iris-virginica":
                attempted = (0, 0, 1)
            inData.append(list((inValues, attempted)))
aa = inData.pop(123)
#print(inData)

np.random.seed(0)
network = Network(size)
network.train(inData, niter=250, alpha=0.02, decay=0.00)
print("no decay", aa[0], network.feedforward(aa[0]), aa[1])

np.random.seed(0)
network = Network(size)
network.train(inData, niter=250, alpha=0.02, decay=0.9)
print("decay", aa[0], network.feedforward(aa[0]), aa[1])