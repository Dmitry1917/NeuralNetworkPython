import numpy as np
import random

from typing import Protocol
from abc import abstractmethod

#import warnings
#warnings.filterwarnings("error")

## Cost functions, other than square.
## Different activations for different layers. Sigmoid, tanh, linear, ReLU, softmax, softsign.
## L1 regularization.
## L2 regularization.
## Learning rate (eta) changing over time/conditions.
# Layer replaceability.
# Autoencoder.
# Analize gradients during training.

class Cost(Protocol):
    @abstractmethod
    def delta(output, desired, z, activation) -> [float]:
        raise NotImplementedError

    @abstractmethod
    def cost(output, desired) -> float:
        raise NotImplementedError

class CostSquare(Cost):
    @staticmethod
    def delta(output, desired, z, activation):
        return (output - desired) * activation.derivative(z)

    @staticmethod
    def cost(output, desired):
        return 0.5 * np.sum(np.square(output - desired))

class CostCrossEntropy(Cost):
    @staticmethod
    def delta(output, desired, z, activation):
        return output - desired

    @staticmethod
    def cost(output, desired):
        return np.sum(np.nan_to_num(-desired * np.log(output) - (1 - desired) * np.log(1 - output)))

class CostLogLikehood(Cost):
    @staticmethod
    def delta(output, desired, z, activation):
        return output - desired

    @staticmethod
    def cost(output, desired):
        indexOfMax = np.argmax(desired)
        return -np.nan_to_num(np.log(output[indexOfMax][0]))

class ActivationSigmoid:
    @staticmethod
    def activation(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def derivative(z):
        activation = ActivationSigmoid.activation(z)
        return activation * (1 - activation)

class ActivationTanh:
    @staticmethod
    def activation(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        activation = ActivationTanh.activation(z)
        return 1 - activation ** 2

class ActivationLinear:
    @staticmethod
    def activation(z):
        return z

    @staticmethod
    def derivative(z):
        return 1

class ActivationReLU:
    @staticmethod
    def activation(z):
        arr = z.copy()
        arr[arr < 0] = 0
        return arr

    @staticmethod
    def derivative(z):
        arr = z.copy()
        arr[arr < 0] = 0
        arr[arr > 0] = 1
        return arr

class ActivationSoftmax:
    @staticmethod
    def activation(z):
        expSum = np.sum(np.exp(z))
        return np.exp(z) / expSum

    @staticmethod
    def derivative(z):
        # It should not ever go here.
        raise NotImplementedError

class ActivationSoftsign:
    @staticmethod
    def activation(z):
        return z / (1.0 + abs(z))

    @staticmethod
    def derivative(z):
        return 1 / ((1 + abs(z)) ** 2)


class Network:
    def __init__(self, sizes, activations, cost):
        self.numberOfLayers = len(sizes)
        self.sizes = sizes
        self.activations = activations
        self.cost = cost
        self.biases = [np.random.normal(size = (y, 1)) for y in sizes[1:]]
        self.weights = [np.random.normal(scale = 1 / (x ** 0.5), size = (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        #self.biases = [np.array([1, 2, 3]), np.array([4])]
        #self.weights = [np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8, 9]])]

        self.l1R = 0.0
        self.l2R = 0.0
        self.nIEL = 0
        self.lRDL = 0.0

    def feedforward(self, a):
        for w, b, af in zip(self.weights, self.biases, self.activations):
            #originalA = a
            #a = np.dot(w, a) + b
            a = af.activation(np.dot(w, a) + b)
            #print(f"input is {originalA}\nw is {w}\nb is {b}\na is {a}")
        return a

    def sgd(self, trainingData, maxEpochs, batchSize, eta, testData = None, evalByMaxElement = False, evalByTrainingData = False):
        if testData: numberOfTests = len(testData)
        numberOfTrainingData = len(trainingData)

        lRCD = 1.0
        lastImprovementEpoch = 0
        lastBestEvaluation = 0

        for i in range(maxEpochs):
            random.shuffle(trainingData)
            batches = [trainingData[j:j + batchSize] for j in range(0, numberOfTrainingData, batchSize)]
            for batch in batches:
                self.updateBatch(batch, eta * lRCD)

            if evalByTrainingData:
                evaluationResult = self.evaluate(trainingData, evalByMaxElement)
                if evalByMaxElement:
                    print(f"Epoch {i} training: {evaluationResult[0]} / {numberOfTrainingData} {evaluationResult[0] / numberOfTrainingData}")
                else:
                    print(f"Epoch {i} training: {evaluationResult[0]} in {numberOfTrainingData} tests.")
                print(f"Epoch {i} cost function by training data: {evaluationResult[1]:.10f}")
 
            if testData:
                evaluationResult = self.evaluate(testData, evalByMaxElement)
                if evalByMaxElement:
                    if self.nIEL > 0:
                        if evaluationResult[0] > lastBestEvaluation:
                            lastBestEvaluation = evaluationResult[0]
                            lastImprovementEpoch = i
                        elif i - lastImprovementEpoch > self.nIEL:
                            lastImprovementEpoch = i
                            if lRCD > self.lRDL:
                                lRCD *= 0.5
                                print('Decrease learning rate')
                            else:
                                print('Learning rate is minimized already.')

                    print(f"Epoch {i} test: {evaluationResult[0]} / {numberOfTests} {evaluationResult[0] / numberOfTests}")
                else:
                    print(f"Epoch {i} test: {evaluationResult[0]} in {numberOfTests} tests.")
                print(f"Epoch {i} cost function by test data: {evaluationResult[1]:.10f}")
            else:
                print(f"Epoch {i} completed.")

            print('')

    def updateBatch(self, batch, eta):
        sumOfDeltasForWeights = [np.zeros(w.shape) for w in self.weights]
        sumOfDeltasForBiases = [np.zeros(b.shape) for b in self.biases]
        batchSize = len(batch)

        for input, result in batch:
            deltaWeights, deltaBiases = self.backpropagation(input, result)
            sumOfDeltasForWeights = [dW + dWBatch for dW, dWBatch in zip(sumOfDeltasForWeights, deltaWeights)]
            sumOfDeltasForBiases = [dB + dBBatch for dB, dBBatch in zip(sumOfDeltasForBiases, deltaBiases)]

        # Didn't add L1 regularization fine tuning, like checking for resulting change not pushing over zero - only to it.
        self.weights = [w - eta * (dW / batchSize + self.l2R * w + self.l1R * np.sign(w)) for w, dW in zip(self.weights, sumOfDeltasForWeights)]
        self.biases = [b - eta * dB / batchSize for b, dB in zip(self.biases, sumOfDeltasForBiases)]

    def backpropagation(self, input, result):
        deltaW = [np.zeros(w.shape) for w in self.weights]
        deltaB = [np.zeros(b.shape) for b in self.biases]
        activation = input
        activations = [input]
        zVectorsByLayer = []

        for w, b, af in zip(self.weights, self.biases, self.activations):
            z = np.dot(w, activation) + b
            zVectorsByLayer.append(z)
            activation = af.activation(z)
            activations.append(activation)

        delta = self.cost.delta(activations[-1], result, zVectorsByLayer[-1], self.activations[-1])
        deltaW[-1] = np.dot(delta, activations[-2].transpose())
        deltaB[-1] = delta

        for i in range(2, self.numberOfLayers):
            z = zVectorsByLayer[-i]
            aD = self.activations[-i].derivative(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * aD
            deltaW[-i] = np.dot(delta, activations[-i-1].transpose())
            deltaB[-i] = delta

        return (deltaW, deltaB)

    def evaluate(self, testData, evalByMaxElement):
        testDataLen = len(testData)
        tests = [(self.feedforward(x), y) for x, y in testData]
        costSum = 0
        diffBelow05 = []
        for i, (real, desired) in enumerate(tests):
            #netResult = self.feedforward(testData[i][0])
            #squareDiff = np.mean(np.square(testData[i][1] - netResult))
            costSum += self.cost.cost(real, desired)
            #print(f"Sample {i}: {testData[i][0]} result {real} expected {desired}")

            # Maybe remove it at all, leaving only cost function for not max?
            if not evalByMaxElement:
                diff = desired - real
                absDiff = abs(diff.reshape(-1))
                diffBelow05.append((sum((x < 0.5) for x in absDiff), len(real)))

        if evalByMaxElement:
            #testResults = [(np.argmax(real), np.argmax(desired)) for real, desired in tests]
            testResults = [(np.argmax(real), np.argmax(desired)) for real, desired in tests]
            successes = sum(int(x == y) for x, y in testResults)
            ##print(testResults)
        else:
            successes = diffBelow05

        cost = costSum / testDataLen

        if self.l1R > 0 or self.l2R > 0:
            # All arrays in weights are of different dimensions, thus simple ways dont work.
            weights = []
            for nparr in self.weights:
                weights.extend(nparr.reshape(-1).tolist())
            weights = np.array(weights)

        if self.l1R > 0:
            cost += self.l1R * np.sum(abs(weights))

        if self.l2R > 0:
            cost += self.l2R * np.sum(weights ** 2) * 0.5

        return (successes, cost)

def loadMNIST(fileName):
    with open(fileName, "rb") as file:
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
#net = Network([2, 2, 1], [ActivationSoftsign(), ActivationSigmoid()], CostCrossEntropy())
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
trainLabelsVectorized = [np.array([np.reshape(x, (1)) for x in vectorized(trainLabels[1][i], 10)]) for i in range(numberOfTrainSamples)]
#print(trainLabelsVectorized[2])

normalizedImages = [i/256.0 for i in testImages[1]]
splitedTestImages = [normalizedImages[i*resolution:i*resolution + resolution] for i in range(testImages[0][0])]
splitedTestImages = [np.reshape(x, (resolution, 1)) for x in splitedTestImages]
#for i in range(testImages[0][1]):
#    print(splitedTestImages[1][i * testImages[0][2]:(i + 1) * testImages[0][2]])
#
testLabelsVectorized = [np.array([np.reshape(x, (1)) for x in vectorized(testLabels[1][i], 10)]) for i in range(numberOfTestSamples)]
#print(testLabelsVectorized[1])
#print(np.array(testLabelsVectorized[1]))
#exit()

trainData = list(zip(splitedTrainImages, trainLabelsVectorized))
testData = list(zip(splitedTestImages, testLabelsVectorized))

mnistNetwork = Network([resolution, 30, 10], [ActivationSigmoid(), ActivationSoftmax()], CostLogLikehood())#CostSquare())
maxEpochs = 30
batchSize = 10
eta = 0.5

mnistNetwork.l1R = 0.00005
mnistNetwork.l2R = 0.00001
mnistNetwork.nIEL = 1
mnistNetwork.lRDL = 0.03125

mnistNetwork.sgd(trainData, maxEpochs, batchSize, eta, testData, evalByMaxElement = True, evalByTrainingData = True)

