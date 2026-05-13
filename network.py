import numpy as np
import random

from typing import Protocol
from abc import abstractmethod

import matplotlib.pyplot as plt
from scipy import ndimage

#import warnings
#warnings.filterwarnings("error")

## Cost functions, other than square.
## Different activations for different layers. Sigmoid, tanh, linear, ReLU, softmax, softsign.
## L1 regularization.
## L2 regularization.
## Learning rate (eta) changing over time/conditions.
## Layer replaceability and individual training.
## Autoencoder.
## Analize gradients during training (per layer).
## Momentum.

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

def weightsInit(neurons, weightsPerNeuron):
    return np.random.normal(scale = 1 / (weightsPerNeuron ** 0.5), size = (neurons, weightsPerNeuron))

def biasesInit(neurons):
    return np.random.normal(size = (neurons, 1))

class Network:
    def __init__(self, sizes, activations, cost):
        self.numberOfLayers = len(sizes)
        self.sizes = sizes
        self.activations = activations
        self.cost = cost
        self.biases = [biasesInit(y) for y in sizes[1:]]
        self.weights = [weightsInit(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.trainability = [True] * (self.numberOfLayers - 1)
        self.l1R = 0.0
        self.l2R = 0.0
        self.nIEL = 0
        self.lRDL = 0.0
        self.momentum = 0.0

        self.gradientsAnalysis = False
        self.l1RLayerDependent = False

    def feedforward(self, a):
        for w, b, af in zip(self.weights, self.biases, self.activations):
            a = af.activation(np.dot(w, a) + b)
        return a

    def showGradientsStatistics(self, gradientsPerEpochs, weightsPerEpochs):
        epochs = range(len(gradientsPerEpochs))
        for i in range(self.numberOfLayers - 1):
            meanGradientAbs = [np.mean(abs(layers[i])) for layers in gradientsPerEpochs]
            plt.subplot(2, 1, 1)
            plt.plot(epochs, meanGradientAbs, label = f'{i}')

            plt.annotate(f'{meanGradientAbs[0]:.7f}',
            (0, meanGradientAbs[0]))
            plt.annotate(f'{meanGradientAbs[-1]:.7f}',
            (len(epochs) - 1, meanGradientAbs[-1]))

            meanWeightsAbs = [np.mean(abs(layers[i])) for layers in weightsPerEpochs]
            plt.subplot(2, 1, 2)
            plt.plot(epochs, meanWeightsAbs, label = f'{i}')

            plt.annotate(f'{meanWeightsAbs[0]:.7f}',
            (0, meanWeightsAbs[0]))
            plt.annotate(f'{meanWeightsAbs[-1]:.7f}',
            (len(epochs) - 1, meanWeightsAbs[-1]))

        plt.subplot(2, 1, 1)
        plt.title('Gradients')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.title('Weights')
        plt.legend()

        plt.show()

    def sgd(self,
            trainingData,
            maxEpochs,
            batchSize,
            eta,
            testData = None,
            evalByMaxElement = False,
            evalByTrainingData = False,
            deformMNISTRandomlyBetweenEpochs = False,
            learningRateChangeByEpoch = (0, 0)):

        if testData: numberOfTests = len(testData)
        numberOfTrainingData = len(trainingData)

        lRCD = 1.0
        lastImprovementEpoch = 0
        lastBestEvaluation = 0

        gradientsPerEpochs = []
        weightsPerEpochs = []

        weightsVelocities = [np.zeros(w.shape) for w in self.weights]
        biasesVelocities = [np.zeros(b.shape) for b in self.biases]

        for i in range(maxEpochs):
            random.shuffle(trainingData)
            actualTrainingData = trainingData

            if deformMNISTRandomlyBetweenEpochs:
                actualTrainingData = []
                for data in trainingData:
                    angle = np.random.uniform(-15, 15)
                    imageArray = data[0].reshape(28, 28)
                    imageArray = ndimage.rotate(imageArray, angle = angle, reshape = False, order = 3)

                    zoomX = np.random.uniform(0.85, 1.15)
                    zoomY = np.random.uniform(0.85, 1.15)

                    # Inverted multipliers.
                    hor = 1 / zoomX
                    ver = 1 / zoomY
                    matrix = np.array([[ver, 0.0], [0.0, hor]])
                    offsets = np.array([28, 28]) * np.array([zoomY - 1, zoomX - 1]) * 0.5
                    imageArray = ndimage.affine_transform(imageArray, matrix = matrix, offset = offsets, output_shape = (28, 28))

                    actualTrainingData.append((imageArray.reshape(784, 1), data[1]))

            batches = [actualTrainingData[j:j + batchSize] for j in range(0, numberOfTrainingData, batchSize)]

            gradients = [np.zeros(w.shape) for w in self.weights]

            if self.gradientsAnalysis:
                weightsPerEpochs.append(self.weights.copy())

            for batch in batches:
                self.updateBatch(batch, eta * lRCD, gradients, weightsVelocities, biasesVelocities)

            if self.gradientsAnalysis:
                gradientsPerEpochs.append([gradientsSumsByLayer / len(batches) for gradientsSumsByLayer in gradients])

            if evalByTrainingData:
                evaluationResult = self.evaluate(actualTrainingData, evalByMaxElement)
                if evalByMaxElement:
                    print(f"Epoch {i} training: {evaluationResult[0]} / {numberOfTrainingData} {evaluationResult[0] / numberOfTrainingData}")
                print(f"Epoch {i} cost function by training data: {evaluationResult[1]:.10f}")

            if learningRateChangeByEpoch[0] > 0:
                print(f'Current learning rate: {eta * lRCD}')
                if learningRateChangeByEpoch[1] < lRCD:
                    lRCD *= learningRateChangeByEpoch[0]

            if testData:
                evaluationResult = self.evaluate(testData, evalByMaxElement)
                if evalByMaxElement:
                    if self.nIEL > 0 and learningRateChangeByEpoch[0] == 0:
                        if evaluationResult[0] > lastBestEvaluation:
                            lastBestEvaluation = evaluationResult[0]
                            lastImprovementEpoch = i
                        elif i - lastImprovementEpoch > self.nIEL:
                            lastImprovementEpoch = i
                            # Prevent constant lR decrease, caused by early accident spike in results.
                            lastBestEvaluation = evaluationResult[0]
                            if lRCD > self.lRDL:
                                lRCD *= 0.5
                                print('Decrease learning rate')
                            else:
                                print('Learning rate is minimized already.')

                    print(f"Epoch {i} test: {evaluationResult[0]} / {numberOfTests} {evaluationResult[0] / numberOfTests}")
                print(f"Epoch {i} cost function by test data: {evaluationResult[1]:.10f}")
            else:
                print(f"Epoch {i} completed.")

            print('')

        if self.gradientsAnalysis:
            self.showGradientsStatistics(gradientsPerEpochs, weightsPerEpochs)

    def updateBatch(self, batch, eta, gradients, weightsVelocities, biasesVelocities):
        sumOfDeltasForWeights = [np.zeros(w.shape) for w in self.weights]
        sumOfDeltasForBiases = [np.zeros(b.shape) for b in self.biases]
        batchSize = len(batch)

        for input, result in batch:
            deltaWeights, deltaBiases = self.backpropagation(input, result)
            sumOfDeltasForWeights = [dW + dWBatch for dW, dWBatch in zip(sumOfDeltasForWeights, deltaWeights)]
            sumOfDeltasForBiases = [dB + dBBatch for dB, dBBatch in zip(sumOfDeltasForBiases, deltaBiases)]

        # Didn't add L1 regularization fine tuning, like checking for resulting change not pushing over zero - only to it.
        #self.weights = [w - eta * (dW / batchSize + self.l2R * w + self.l1R * np.sign(w)) for w, dW in zip(self.weights, sumOfDeltasForWeights)]
        #self.biases = [b - eta * dB / batchSize for b, dB in zip(self.biases, sumOfDeltasForBiases)]

        for trainability, w, dW, b, dB, index in zip(self.trainability, self.weights, sumOfDeltasForWeights, self.biases, sumOfDeltasForBiases, range(self.numberOfLayers - 1)):
            if trainability:
                l1R = self.l1R
                if self.l1RLayerDependent:
                    # Coefficients are results of testing, to make L1 to gradient ratio relatively same between layers.
                    lastIndex = self.numberOfLayers - 2
                    if index < lastIndex:
                        if index == 0:
                            l1R *= 0.2
                        else:
                            l1R *= 0.5# * (0.4 + 0.6 * index / (lastIndex - 1))# Will be there only if lastIndex > 1.

                gradientPure = dW / batchSize
                if self.gradientsAnalysis:
                    gradients[index] = gradients[index] + gradientPure

                weightGradient = gradientPure + self.l2R * w + l1R * np.sign(w)
                weightVelocity = weightGradient

                biasGradient = dB / batchSize
                biasVelocity = biasGradient

                if self.momentum > 0:
                    weightVelocity = self.momentum * weightsVelocities[index] + (1 - self.momentum) * weightVelocity
                    biasVelocity = self.momentum * biasesVelocities[index] + (1 - self.momentum) * biasVelocity
                    weightsVelocities[index] = weightVelocity
                    biasesVelocities[index] = biasVelocity

                self.weights[index] = w - eta * weightVelocity
                self.biases[index] = b - eta * biasVelocity

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
            # Trainability will be used only for blocking first layers from training, thefore backprop can be stoped for better performance.
            if self.trainability[-i]:
                z = zVectorsByLayer[-i]
                aD = self.activations[-i].derivative(z)
                delta = np.dot(self.weights[-i+1].transpose(), delta) * aD
#                if np.isnan(aD).any() or np.isnan(delta).any() or np.isnan(self.weights[-i+1]).any():
#                    print(f'delta {delta}\nweights {self.weights[-i+1]}\naD {aD}')
#                    exit()
                deltaW[-i] = np.dot(delta, activations[-i-1].transpose())
                deltaB[-i] = delta

        return (deltaW, deltaB)

    def evaluate(self, testData, evalByMaxElement):
        testDataLen = len(testData)
        tests = [(self.feedforward(x), y) for x, y in testData]
        costSum = 0
        diffBelow05 = []
        for i, (real, desired) in enumerate(tests):
            costSum += self.cost.cost(real, desired)

        if evalByMaxElement:
            testResults = [(np.argmax(real), np.argmax(desired)) for real, desired in tests]
            successes = sum(int(x == y) for x, y in testResults)
        else:
            successes = []

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

    def addHiddenLayer(self, size, activationFunction):
        self.numberOfLayers += 1
        self.sizes.insert(-1, size)
        self.trainability.insert(-1, True)
        self.activations.insert(-1, activationFunction)
        # Added hidden layer can have other size, then previous last hidden, thus last layer weights must be replaced.
        self.weights.pop()
        self.biases.pop()

        self.biases.append(biasesInit(self.sizes[-2]))
        self.weights.append(weightsInit(self.sizes[-2], self.sizes[-3]))

        self.biases.append(biasesInit(self.sizes[-1]))
        self.weights.append(weightsInit(self.sizes[-1], self.sizes[-2]))

    def autoencoderPretrain(self, trainingData, maxEpochs, batchSize, eta):
        lastLayerWeights = self.weights.pop()
        lastLayerBiases = self.biases.pop()
        self.biases.append(biasesInit(self.sizes[0]))
        self.weights.append(weightsInit(self.sizes[0], self.sizes[-2]))

        originalCost = self.cost
        self.cost = CostSquare()
        originalLastActivation = self.activations.pop()
        self.activations.append(ActivationLinear())

        print('Start pretraining.\n')
        self.sgd(trainingData, maxEpochs, batchSize, eta, evalByTrainingData = True)

        # Draw results of autoencoder for comparison.
        #imagesCount = 10
        #for imageIndex in range(imagesCount):
        #    image = trainingData[imageIndex]
        #    imageArray = image[0].reshape(28, 28)

        #    plt.subplot(imagesCount, 2, imageIndex * 2 + 1)
        #    plt.imshow(imageArray, cmap='gray')
        #    imageArray = self.feedforward(image[0]).reshape(28, 28)

        #    plt.subplot(imagesCount, 2, imageIndex * 2 + 2)
        #    plt.imshow(imageArray, cmap='gray')

        #plt.show()

        # Restore changes in network.
        self.activations.pop()
        self.activations.append(originalLastActivation)
        self.cost = originalCost
        self.weights.pop()
        self.weights.append(lastLayerWeights)
        self.biases.pop()
        self.biases.append(lastLayerBiases)

        # Train only output layer afterwards.
        self.trainability = [False] * (self.numberOfLayers - 2)
        self.trainability.append(True)

        print('End pretraining.\n')

def loadMNIST(fileName):
    with open(fileName, "rb") as file:
        mainInfoBuffer = file.read(4)
        dimensionsAmount = mainInfoBuffer[3]
        if dimensionsAmount > 0:
            dimensionsBufferSize = dimensionsAmount * 4
            sampleSize = 1
            dimensions = []
            for i in range(dimensionsAmount):
                dimensionBytes = file.read(4)
                dimensions.append(int.from_bytes(dimensionBytes, byteorder = 'big'))

            samplesCount = dimensions[0]
            if dimensionsAmount > 1:
                for i in range(1, dimensionsAmount):
                    sampleSize *= dimensions[i]
            samples = file.read(samplesCount * sampleSize)
            return (dimensions, samples)
        else:
            print(f"Wrong MNIST file format {filename}")

def vectorized(i, n):
    vector = np.zeros(n, dtype = 'int')
    vector[i] = 1
    return vector

# XOR example
#net = Network([2, 2, 1], [ActivationSoftsign(), ActivationSigmoid()], CostCrossEntropy())
#inputsRaw = [[0, 0], [0, 1], [1, 0], [1, 1]]
#inputs = [np.reshape(x, (2, 1)) for x in inputsRaw]
#outputsXOR = [np.reshape(x, (1, 1)) for x in [0, 1, 1, 0]]
#trainDataXOR = list(zip(inputs, outputsXOR))
#net.sgd(trainDataXOR, 400, 2, 3.0, trainDataXOR)
#exit()

# Regression example
#net = Network([1, 40, 40, 1], [ActivationReLU(), ActivationReLU(), ActivationLinear()], CostSquare())
#def operation(x):
#    return 3 * x**3 - x**2 + 7 * x + 5
#
#inputsRaw = np.arange(-25, 25, 0.2)
#maxInputRaw = np.max(np.abs(inputsRaw))
#inputs = [np.reshape(x / maxInputRaw, (1, 1)) for x in inputsRaw]
#outputsRaw = [operation(x) for x in inputsRaw]
#maxOutputRaw = np.max(np.abs(outputsRaw))
#outputs = [np.reshape(x / maxOutputRaw, (1, 1)) for x in outputsRaw]
##plt.scatter(inputsRaw, outputsRaw)
##plt.show()
#
#trainData = list(zip(inputs, outputs))
#inputsTestRaw = np.arange(-30, 30, 0.5)
#inputsTest = [np.reshape(x / maxInputRaw, (1, 1)) for x in inputsTestRaw]
#outputsTestRaw = [operation(x) for x in inputsTestRaw]
#outputsTest = [np.reshape(x / maxOutputRaw, (1, 1)) for x in outputsTestRaw]
#testData = list(zip(inputsTest, outputsTest))
#
#net.sgd(trainData, 1000, 10, 0.01, testData, evalByTrainingData = True)
#
#outputsNet = [net.feedforward(x).item() * maxOutputRaw for x in inputsTest]
#plt.scatter(inputsTestRaw, outputsTestRaw)
#plt.scatter(inputsTestRaw, outputsNet)
#plt.show()
#
#exit()

# MNIST example
trainImages = loadMNIST("train-images.idx3-ubyte")
trainLabels = loadMNIST("train-labels.idx1-ubyte")
testImages = loadMNIST("t10k-images.idx3-ubyte")
testLabels = loadMNIST("t10k-labels.idx1-ubyte")

numberOfTrainSamples = trainImages[0][0]
numberOfTestSamples = testImages[0][0]

# Prepare bare data for analisis.
resolution = trainImages[0][1] * trainImages[0][2]
normalizedImages = [i/256.0 for i in trainImages[1]]
splitedTrainImages = [normalizedImages[i*resolution:i*resolution + resolution] for i in range(trainImages[0][0])]
splitedTrainImages = [np.reshape(x, (resolution, 1)) for x in splitedTrainImages]

trainLabelsVectorized = [np.array([np.reshape(x, (1)) for x in vectorized(trainLabels[1][i], 10)]) for i in range(numberOfTrainSamples)]

normalizedImages = [i/256.0 for i in testImages[1]]
splitedTestImages = [normalizedImages[i*resolution:i*resolution + resolution] for i in range(testImages[0][0])]
splitedTestImages = [np.reshape(x, (resolution, 1)) for x in splitedTestImages]

testLabelsVectorized = [np.array([np.reshape(x, (1)) for x in vectorized(testLabels[1][i], 10)]) for i in range(numberOfTestSamples)]

trainData = list(zip(splitedTrainImages, trainLabelsVectorized))
testData = list(zip(splitedTestImages, testLabelsVectorized))

# Try image rotation.
#imagesCount = 10
#for imageIndex in range(imagesCount):
#    image = trainData[imageIndex]
#    imageArray = image[0].reshape(28, 28)
#
#    plt.subplot(imagesCount, 2, imageIndex * 2 + 1)
#    plt.imshow(imageArray, cmap='gray')
#
#    angle = np.random.uniform(-15, 15)
#    #print(angle)
#    imageArray = ndimage.rotate(imageArray, angle = angle, reshape = False, order = 3)
#
#    zoomX = np.random.uniform(0.85, 1.15)
#    zoomY = np.random.uniform(0.85, 1.15)
#    #print(zoomX, zoomY)
#    # Inverted multipliers.
#    hor = 1 / zoomX
#    ver = 1 / zoomY
#    matrix = np.array([[ver, 0.0], [0.0, hor]])
#    offsets = np.array([28, 28]) * np.array([zoomY - 1, zoomX - 1]) * 0.5
#    imageArray = ndimage.affine_transform(imageArray, matrix = matrix, offset = offsets, output_shape = (28, 28))
#
#    plt.subplot(imagesCount, 2, imageIndex * 2 + 2)
#    plt.imshow(imageArray, cmap='gray')
#
#plt.show()
#exit()



mnistNetwork = Network([resolution, 30, 10], [ActivationSigmoid(), ActivationSoftmax()], CostLogLikehood())#CostSquare())
maxEpochs = 30
batchSize = 10
eta = 1.00

#mnistNetwork.l1R = 0.00005

#autoencoderTrainData = list(zip(splitedTrainImages, splitedTrainImages))
#mnistNetwork.autoencoderPretrain(autoencoderTrainData, maxEpochs = 10, batchSize = batchSize, eta = 0.005)
#mnistNetwork.addHiddenLayer(30, ActivationTanh())
#mnistNetwork.autoencoderPretrain(autoencoderTrainData, maxEpochs = 10, batchSize = batchSize, eta = 0.005)
#mnistNetwork.addHiddenLayer(30, ActivationTanh())
#mnistNetwork.autoencoderPretrain(autoencoderTrainData, maxEpochs = 10, batchSize = batchSize, eta = 0.005)
#mnistNetwork.addHiddenLayer(30, ActivationTanh())
#mnistNetwork.autoencoderPretrain(autoencoderTrainData, maxEpochs = 10, batchSize = batchSize, eta = 0.005)

#mnistNetwork.trainability = [True] * (mnistNetwork.numberOfLayers - 1)

mnistNetwork.l1R = 0.00005
mnistNetwork.l2R = 0.00001
mnistNetwork.nIEL = 1
mnistNetwork.lRDL = 0.015625

mnistNetwork.momentum = 0.9
#mnistNetwork.l1RLayerDependent = True
#mnistNetwork.gradientsAnalysis = True
mnistNetwork.sgd(
        trainData,
        maxEpochs,
        batchSize,
        eta,
        testData,
        evalByMaxElement = True,
        evalByTrainingData = True,
#        deformMNISTRandomlyBetweenEpochs = True,
        learningRateChangeByEpoch = (0.9, 0.01)
)

