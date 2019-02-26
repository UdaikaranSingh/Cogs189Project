from scipy.io import loadmat
import numpy as np
import pandas as pd

def getData(data):
	samplingRate = data['data'][0][0][0][0][0][0]
	numChannels = data['data'][0][0][1][0][0][0]
	timePointLabels = data['data'][0][0][3][0][0]
	rawSignal = data['data'][0][0][4][0]
	timeOfEventandLabel = data['data'][2][0]
	return (samplingRate, numChannels, timePointLabels, rawSignal, timeOfEventandLabel)

def createSamples(x,y, samplingRate):
    timetoLabel = pd.DataFrame(y)
    timetoLabel['samples'] = timetoLabel[0] * samplingRate
    count = 0
    X_train = []
    labels = []
    
    maxTime = timetoLabel.samples.max()
    
    labeltoIndex = [10.0, 1.0, 7.0, 3.0, 8.0, 4.0]
    
    while (count + 1) * 500 < maxTime:
        
        median = (500 * count + 500 * (count + 1)) / 2        
        labelVal = timetoLabel[timetoLabel.samples > median].iloc[0][1]
        if labelVal >= 0:
            sample = x[500 * count: 500 * (count + 1),:]
            X_train.append(sample)


            labelVal = timetoLabel[timetoLabel.samples > median].iloc[0][1]

            label = np.zeros(6)
            label[labeltoIndex.index(labelVal)] = 1

            labels.append(label)
        count = count + 1
    
    return np.array(X_train).astype('float32'), np.array(labels).astype('float32')

def getAllData(lstofFiles):
	path = "./data"
	X_trains = []
	labels = []
	for i in lstofFiles:
		matFile = loadmat(path + '/' + i)
		samplingRate, numChannels, timePointLabels, x, y = getData(matFile)
		X_train, label = createSamples(x= x, y = y, samplingRate=samplingRate)
		X_trains.append(X_train)
		labels.append(label)
	return np.concatenate(X_trains, axis = 0), np.concatenate(labels, axis = 0)