import numpy as np
from pybann import Model

def check(code):
    if code == (1, 0, 0):
        return "Iris-setosa"
    if code == (0, 1, 0):
        return "Iris-versicolor"
    if code == (0, 0, 1):
        return "Iris-virginica"

#import network
# Initialize network
#np.random.seed(10)
network = Model(name='IRIS example')

network.addInput(neurons=4)
network.addLayer(neurons=9, activation="sigmoid")
network.addLayer(neurons=9, activation="sigmoid")
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

network.SGD(dataset=inData, batchsize=25, alpha=1.e-3, momentum=0.90, nepoch=10000)

loss = 0.
for i in range(len(inDataTest)):
    result = network.forward(inValues=inDataTest[i][0])
    loss += np.sum((inDataTest[i][1]-result)**2)

loss /= float(len(inDataTest))

print('LOSS:: ', loss, (1-loss)*100)

# Example
size = 40
success_counter = 0
index_list = np.random.randint(len(inData), size=size)

for index in index_list:
    result = network.forward(inValues=inData[index][0])
    for i in range(len(result)):
        if result[i] == np.amax(result):
            result[i] = 1
        else:
            result[i] = 0
    print("DATA::", inData[index], check(inData[index][1]))
    print("RESULT::", tuple(result), check(tuple(result)))
    if check(inData[index][1]) == check(tuple(result)):
        print("SUCCES")
        success_counter += 1
    else:
        print("FAIL")
    print("---")

print("SUCCESS::", success_counter, "/", size)
#print("EXAMPLE::", inData[42])
#results = network.forward(inValues=inData[42][0])
#print("RESULT::", network.forward(inValues=inData[42][0])
#print("* ", results, inData[42][1])