import numpy as np
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def activationDerivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network:
    def __init__(self, sizes):
        self.numberOfLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.normal(size=(y, 1)) for y in sizes[1:]]
        self.weights = [np.random.normal(size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        #self.biases = [np.array([1, 2, 3]), np.array([4])]
        #self.weights = [np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8, 9]])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            #originalA = a
            #a = np.dot(w, a) + b
            a = sigmoid(np.dot(w, a) + b)
            #print(f"input is {originalA}\nw is {w}\nb is {b}\na is {a}")
        return a

    def sgd(self, trainingData, maxEpochs, batchSize, eta, testData = None, evalByMaxElement = False):
        if testData: numberOfTests = len(testData)
        numberOfTrainingData = len(trainingData)
        for i in range(maxEpochs):
            random.shuffle(trainingData)
            batches = [trainingData[j:j + batchSize] for j in range(0, numberOfTrainingData, batchSize)]
            for batch in batches:
                self.updateBatch(batch, eta)
            if testData:
                evaluationResult = self.evaluate(testData, evalByMaxElement)
                if evalByMaxElement:
                    print(f"Epoch {i}: {evaluationResult[0]} / {numberOfTests}")
                else:
                    print(f"Epoch {i}: {evaluationResult[0]} in {numberOfTests} tests.")
                print(f"Epoch {i} cost function by test data: {evaluationResult[1]:.10f}")
            else:
                print(f"Epoch {i} completed.")

    def updateBatch(self, batch, eta):
        sumOfDeltasForWeights = [np.zeros(w.shape) for w in self.weights]
        sumOfDeltasForBiases = [np.zeros(b.shape) for b in self.biases]
        batchSize = len(batch)

        for input, result in batch:
            deltaWeights, deltaBiases = self.backpropagation(input, result)
            sumOfDeltasForWeights = [dW + dWBatch for dW, dWBatch in zip(sumOfDeltasForWeights, deltaWeights)]
            sumOfDeltasForBiases = [dB + dBBatch for dB, dBBatch in zip(sumOfDeltasForBiases, deltaBiases)]

        self.weights = [w - eta * dW / batchSize for w, dW in zip(self.weights, sumOfDeltasForWeights)]
        self.biases = [b - eta * dB / batchSize for b, dB in zip(self.biases, sumOfDeltasForBiases)]

    def backpropagation(self, input, result):
        deltaW = [np.zeros(w.shape) for w in self.weights]
        deltaB = [np.zeros(b.shape) for b in self.biases]
        activation = input
        activations = [input]
        zVectorsByLayer = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zVectorsByLayer.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.costDerivative(activations[-1], result) * activationDerivative(zVectorsByLayer[-1])
        deltaW[-1] = np.dot(delta, activations[-2].transpose())
        deltaB[-1] = delta

        for i in range(2, self.numberOfLayers):
            z = zVectorsByLayer[-i]
            aD = activationDerivative(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * aD
            deltaW[-i] = np.dot(delta, activations[-i-1].transpose())
            deltaB[-i] = delta

        return (deltaW, deltaB)

    def evaluate(self, testData, evalByMaxElement):
        testDataLen = len(testData)
        tests = [(self.feedforward(x), y) for x, y in testData]
        squareDiffSum = 0
        diffBelow05 = []
        for i, (real, desired) in enumerate(tests):
            #netResult = self.feedforward(testData[i][0])
            #squareDiff = np.mean(np.square(testData[i][1] - netResult))
            diff = desired - real
            squareDiffSum += np.mean(np.square(diff))
            #print(f"Sample {i}: {testData[i][0]} result {real} expected {desired}")
            if not evalByMaxElement:
                absDiff = abs(diff.reshape(-1))
                diffBelow05.append((sum((x < 0.5) for x in absDiff), len(real)))

        if evalByMaxElement:
            #testResults = [(np.argmax(real), np.argmax(desired)) for real, desired in tests]
            testResults = [(np.argmax(real), np.argmax(desired)) for real, desired in tests]
            successes = sum(int(x == y) for x, y in testResults)
            ##print(testResults)
        else:
            successes = diffBelow05

        return (successes, squareDiffSum / testDataLen)

    def costDerivative(self, output, desired):
        return output - desired

def loadMNIST(fileName):
    file = open(fileName, "rb")
    mainInfoBuffer = file.read(4)
    #print(mainInfoBuffer)
    dimensionsAmount = mainInfoBuffer[3]
    #print(dimensionsAmount)
    if dimensionsAmount > 0:
        dimensionsBufferSize = dimensionsAmount * 4
        sampleSize = 1
        dimensions = []
        for i in range(dimensionsAmount):
            dimensionBytes = file.read(4)
            dimensions.append(int.from_bytes(dimensionBytes, byteorder = 'big'))
        #dimensionsBuffer = file.read(dimensionsBufferSize)
        #print(dimensionsBuffer)
        #print(len(dimensionsBuffer))
        #print(dimensions)
        samplesCount = dimensions[0]
        if dimensionsAmount > 1:
            for i in range(1, dimensionsAmount):
                sampleSize *= dimensions[i]
        #print(sampleSize)
        samples = file.read(samplesCount * sampleSize)
        return (dimensions, samples)
    else:
        print(f"Wrong MNIST file format {filename}")

def vectorized(i, n):
    vector = np.zeros(n, dtype = 'int')
    vector[i] = 1
    return vector

#net = Network([2, 3, 1])
#print(net.weights)
#print(net.biases)
#print(net.feedforward([1, 1]))

#matrix1 = np.array([[1, 2], [3, 4]])
#vector1 = np.array([2, 2])
#matrix2 = np.array([[1, 1]])
#print(np.dot(matrix1, vector1) + matrix2)

# XOR example
#net = Network([2, 2, 1])
#inputsRaw = [[0, 0], [0, 1], [1, 0], [1, 1]]
#inputs = [np.reshape(x, (2, 1)) for x in inputsRaw]
#outputsXOR = [np.reshape(x, (1, 1)) for x in [0, 1, 1, 0]]
#trainDataXOR = list(zip(inputs, outputsXOR))
#net.sgd(trainDataXOR, 400, 2, 3.0, trainDataXOR)
#exit()

# MNIST example
trainImages = loadMNIST("train-images.idx3-ubyte")
trainLabels = loadMNIST("train-labels.idx1-ubyte")
testImages = loadMNIST("t10k-images.idx3-ubyte")
testLabels = loadMNIST("t10k-labels.idx1-ubyte")
#print(trainImages[0])
#shift = 784 * 2
#for i in range(trainImages[0][1]):
#    print(trainImages[1][shift + i * trainImages[0][2]:shift + (i + 1) * trainImages[0][2]])
#
#print(trainLabels[1][2])

numberOfTrainSamples = trainImages[0][0]
numberOfTestSamples = testImages[0][0]
# Prepare bare data for analisis.
resolution = trainImages[0][1] * trainImages[0][2]
normalizedImages = [i/256.0 for i in trainImages[1]]
splitedTrainImages = [normalizedImages[i*resolution:i*resolution + resolution] for i in range(trainImages[0][0])]
splitedTrainImages = [np.reshape(x, (resolution, 1)) for x in splitedTrainImages]
#for i in range(trainImages[0][1]):
#    print(splitedTrainImages[2][i * trainImages[0][2]:(i + 1) * trainImages[0][2]])
#
trainLabelsVectorized = [[np.reshape(x, (1)) for x in vectorized(trainLabels[1][i], 10)] for i in range(numberOfTrainSamples)]
#print(trainLabelsVectorized[2])

normalizedImages = [i/256.0 for i in testImages[1]]
splitedTestImages = [normalizedImages[i*resolution:i*resolution + resolution] for i in range(testImages[0][0])]
splitedTestImages = [np.reshape(x, (resolution, 1)) for x in splitedTestImages]
#for i in range(testImages[0][1]):
#    print(splitedTestImages[1][i * testImages[0][2]:(i + 1) * testImages[0][2]])
#
testLabelsVectorized = [[np.reshape(x, (1)) for x in vectorized(testLabels[1][i], 10)] for i in range(numberOfTestSamples)]
#print(testLabelsVectorized[1])


trainData = list(zip(splitedTrainImages, trainLabelsVectorized))
testData = list(zip(splitedTestImages, testLabelsVectorized))

mnistNetwork = Network([resolution, 30, 10])
maxEpochs = 10
batchSize = 10
eta = 3.0
mnistNetwork.sgd(trainData, maxEpochs, batchSize, eta, testData, evalByMaxElement = True)

