import numpy as np
from pybann import Model

#import network
# Initialize network
#np.random.seed(10)
network = Model(name='IRIS example')

network.addInput(neurons=8)
network.addLayer(neurons=16, activation="sigmoid")
network.addLayer(neurons=16, activation="sigmoid")
network.addLayer(neurons=16, activation="sigmoid")
network.addLayer(neurons=1, activation="sigmoid")

network.build()

# Read data
with open('data/htru2/HTRU_2.data', 'r') as f:
    lines = f.readlines()
    inData = []
    inDataTest = []
    train = True
    for line in lines:
        line = line.strip().split(",")
        if len(line) > 1:
            inValues = (float(line[0]), float(line[1]), float(line[2]), float(line[3]),float(line[4]), float(line[5]), float(line[6]), float(line[7]))
            if train == True:
                inData.append(list((inValues, float(line[-1]))))
                train = False
            else:
                inDataTest.append(list((inValues, float(line[-1]))))
                train = True

network.SGD(dataset=inData, batchsize=50, alpha=1.e-3, momentum=0.2, nepoch=5000)

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
    result2 = 0 if result <= 0.5 else 1
    print("DATA::", inData[index])
    print("RESULT::", result, result2)
    if inData[index][1] == result2:
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