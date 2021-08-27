import numpy as np
import pickle
from pybann import Model

#import network
# Initialize network
np.random.seed(10)
network = Model(name='IRIS example')

network.addInput(neurons=4)
network.addLayer(neurons=12, activation="sigmoid")
network.addLayer(neurons=3, activation="sigmoid")

network.build()

# Read data
with open('data/iris/iris.data', 'r') as f:
    lines = f.readlines()
    inData = []
    inDataTest = []
    train = True
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
            if train == True:
                inData.append(list((inValues, attempted)))
                train = False
            else:
                inDataTest.append(list((inValues, attempted)))
                train = True

print(len(inData), len(inDataTest))
network.SGD(dataset=inData,alpha=1.e-3, momentum=0.9, niter=2500)

loss = 0.
for i in range(len(inDataTest)):
    result = network.forward(inValues=inDataTest[i][0])
    loss += np.sum((inDataTest[i][1]-result)**2)

loss /= float(len(inDataTest))

print(loss, (1-loss)*100)

network.save("test_network2")
#with open("test_network", "wb") as f:
#    pickle.dump(network, f)

lnet = Model()
lnet.load("test_network2")
print(lnet.layers[1].weights)
#with open("test_network2", "rb") as f:
#    loaded_network = pickle.load(f)