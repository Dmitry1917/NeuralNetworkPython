import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from mnist import loadMNIST
from mnist import vectorized

def weightsInit(neuronsDimensions, weightsPerNeuron):
    return np.random.default_rng().uniform(0.0, 1.0, size = (math.prod(neuronsDimensions), weightsPerNeuron))

class SOM:
    def __init__(self, neuronsDimensions: [int, ...], weightsNumbers):
        self.neuronsDimensions = neuronsDimensions
        self.neurons = weightsInit(neuronsDimensions, weightsNumbers)

    def findBMU(self, sample) -> int:
        squareDistancesNeuronToSample = ((self.neurons - sample) ** 2).sum(axis = 1)
        minIndex = np.argmin(squareDistancesNeuronToSample)
        return minIndex

    def neighbourhood(self, bmuIndex, index, maxRadius):
        if index == bmuIndex:
            return 1.0
        else:
            #1D
            if len(self.neuronsDimensions) == 1:
                diff = abs(index - bmuIndex)
            #2D
            else:
                bmuY = bmuIndex // self.neuronsDimensions[1]
                bmuX = bmuIndex % self.neuronsDimensions[1]
                y = index // self.neuronsDimensions[1]
                x = index % self.neuronsDimensions[1]
                diff = math.sqrt((bmuX - x) ** 2 + (bmuY - y) ** 2)
            if diff < maxRadius or diff < 2:
                return 1.0 / (2 ** diff)
            else:
                return 0

    @staticmethod
    def learningRate(epoch, epochs):
        return 1 - epoch / epochs;

    @staticmethod
    def radiusRate(epoch, epochs):
        return 1 - epoch / epochs;

    def train(self, samples, epochs):
        localSamples = samples.copy()
        maxRadius = math.sqrt(np.dot(self.neuronsDimensions, self.neuronsDimensions))
        for epoch in range(epochs):
            np.random.shuffle(localSamples)
            lR = self.learningRate(epoch, epochs)
            radius = maxRadius * self.radiusRate(epoch, epochs)
            for sample in localSamples:
                bmuIndex = self.findBMU(sample)
                for index, neuron in enumerate(self.neurons):
                    neuron = neuron + self.neighbourhood(bmuIndex, index, radius) * lR * (sample - neuron)
                    self.neurons[index] = neuron

    def print(self):
        #1D
        if len(self.neuronsDimensions) == 1:
            print(self.neurons)
        #2D
        else:
            for i in range(self.neuronsDimensions[0]):
                for k in range(self.neuronsDimensions[1]):
                    print(self.neurons[i * self.neuronsDimensions[1] + k], end = '  ')
                print()

    #Only for 2D network and 2D data.
    def plotPure2D(self):
        #1D
        if len(self.neuronsDimensions) == 1:
            return
        #2D
        #Neurons
        x = []
        y = []
        for neuron in self.neurons:
            x.append(neuron[0])
            y.append(neuron[1])

        plt.title('SOM')
        plt.scatter(x, y)
        plt.show()

    def draw(self, samples = [], labels = []):
        generate2DSpaceRepresentation = len(self.neurons[0]) == 2
        rows = self.neuronsDimensions[0]
        if len(self.neuronsDimensions) == 1:
            columns = 1
        else:
            columns = self.neuronsDimensions[1]

        if generate2DSpaceRepresentation:
            width = 600
            height = width * rows // columns
            img = Image.new('RGB', (2 * width + 100, height), color = 'white')
        else:
            width = 800
            height = width * rows // columns
            img = Image.new('RGB', (width + 50, height), color = 'white')

        draw = ImageDraw.Draw(img, 'RGBA')

# Row/column presentation.
        margin = 20
        neuronAreaWidth = width / columns
        neuronAreaHeight = height / rows
        for i in range(rows):
            for k in range(columns):
                topLeftX = k * neuronAreaWidth + margin / 2
                topLeftY = i * neuronAreaHeight + margin / 2
                bottomRightX = topLeftX + neuronAreaWidth - margin
                bottomRightY = topLeftY + neuronAreaHeight - margin
                draw.rectangle((topLeftX, topLeftY, bottomRightX, bottomRightY), fill = '#22222277')
                draw.text((topLeftX + 3, bottomRightY - 14), f'{i} {k}', fill = '#22ff22')

        labelsInNeurons = [{} for _ in range(len(self.neurons))]

        for index, sample in enumerate(samples):
            bmuIndex = self.findBMU(sample)
            bmuY = bmuIndex // columns
            bmuX = bmuIndex % columns

            if (len(labels) == len(samples)) and len(labels) > 0:
                label = labels[index]
                if label in labelsInNeurons[bmuIndex]:
                    labelsInNeurons[bmuIndex][label] += 1
                else:
                    labelsInNeurons[bmuIndex][label] = 1
            else:
                topLeftX = bmuX * neuronAreaWidth + margin / 2 + np.random.uniform(25)
                topLeftY = bmuY * neuronAreaHeight + margin / 2 + np.random.uniform(25)
                bottomRightX = topLeftX + 2
                bottomRightY = topLeftY + 2
                draw.rectangle((topLeftX, topLeftY, bottomRightX, bottomRightY), fill = '#999999')


        if (len(labels) == len(samples)) and len(labels) > 0:
            for index, labelsInNeuron in enumerate(labelsInNeurons):
                y = index // columns
                x = index % columns
                for index, (label, amount) in enumerate(labelsInNeuron.items()):
                    topLeftX = x * neuronAreaWidth + margin / 2 + index * 10
                    topLeftY = y * neuronAreaHeight + margin / 2 + index * 10
                    draw.text((topLeftX, topLeftY), f'{label} {amount}', fill = '#0000ff')

        if generate2DSpaceRepresentation:
#Value space presentation.
            topLeftXCommon = width + 50
            for index, (neuron, labelsInNeuron) in enumerate(zip(self.neurons, labelsInNeurons)):
                x = 0
                y = 0
                if columns == 1:
                    x = 0.5
                    y = neuron[0]
                else:
                    x = neuron[0]
                    y = neuron[1]
                neuronWidth = neuronAreaWidth * 0.75
                neuronHeight = neuronAreaHeight * 0.75
                topLeftX = topLeftXCommon + x * (width - neuronWidth)
                topLeftY = y * (height - neuronHeight)
                bottomRightX = topLeftX + neuronWidth
                bottomRightY = topLeftY + neuronHeight
                draw.rectangle((topLeftX, topLeftY, bottomRightX, bottomRightY), fill = '#22222277')
                i = index // columns
                k = index % columns
                draw.text((topLeftX + 3, bottomRightY - 14), f'{i} {k}', fill = '#00ff00')
                draw.text((topLeftX + 3, bottomRightY - 30), f'{neuron[0]:.2f} {neuron[1]:.2f}', fill = '#ff0000')
                for index, (label, amount) in enumerate(labelsInNeuron.items()):
                        topLeftX += index * 10
                        topLeftY += index * 10
                        draw.text((topLeftX, topLeftY), f'{label} {amount}', fill = '#0000ff')

        img.show()

def uniformGroups():
    som = SOM((7, 5), 2)
    size = 10
    dataX1 = np.random.uniform(0.0, 0.1, size = size)
    dataY1 = np.random.uniform(0.0, 0.1, size = size)
    
    dataX2 = np.random.uniform(0.0, 0.1, size = size)
    dataY2 = np.random.uniform(0.9, 1.0, size = size)
    
    dataX3 = np.random.uniform(0.9, 1.0, size = size)
    dataY3 = np.random.uniform(0.0, 0.1, size = size)
    
    dataX4 = np.random.uniform(0.9, 1.0, size = size)
    dataY4 = np.random.uniform(0.9, 1.0, size = size)
    
    dataX5 = np.random.uniform(0.45, 0.55, size = size)
    dataY5 = np.random.uniform(0.45, 0.55, size = size)
    
    x = np.concatenate((dataX1, dataX2, dataX3, dataX4, dataX5))
    y = np.concatenate((dataY1, dataY2, dataY3, dataY4, dataY5))
    
    samples = np.stack((x, y), axis = 1)
    
    #print('samples')
    #print(samples)
    
    #som.plotPure2D()
    
    som.train(samples, 100)
    
    #print('final neurons')
    #som.print()
    #som.plotPure2D()
    labels = [f'{i}' for i in range(1, 6) for k in range(size)]
    som.draw(samples, labels)

def normalGroups():
    som = SOM((7, 5), 2)
    size = 10
    dataX1 = np.random.normal(0.2, 0.1, size = size)
    dataY1 = np.random.normal(0.2, 0.1, size = size)
    
    dataX2 = np.random.normal(0.8, 0.1, size = size)
    dataY2 = np.random.normal(0.8, 0.1, size = size)
    
    dataX3 = np.random.normal(0.8, 0.1, size = size)
    dataY3 = np.random.normal(0.2, 0.1, size = size)
    
    dataX4 = np.random.uniform(0.2, 0.1, size = size)
    dataY4 = np.random.uniform(0.8, 0.1, size = size)
    
    dataX5 = np.random.normal(0.5, 0.1, size = size)
    dataY5 = np.random.normal(0.5, 0.1, size = size)
    
    x = np.concatenate((dataX1, dataX2, dataX3, dataX4, dataX5))
    y = np.concatenate((dataY1, dataY2, dataY3, dataY4, dataY5))
    
    samples = np.stack((x, y), axis = 1)
    
    som.train(samples, 500)
    
    labels = [f'{i}' for i in range(1, 6) for k in range(size)]
    som.draw(samples, labels)

def circleGroups():
    som = SOM((20, 15), 2)
    size = 50

    steps = np.linspace(0, 2 * np.pi, size)
    dataX1 = np.array([0.5 + 0.45 * np.cos(angle) for angle in steps])
    dataY1 = np.array([0.5 + 0.45 * np.sin(angle) for angle in steps])
    
    dataX2 = np.array([0.5 + 0.3 * np.cos(angle) for angle in steps])
    dataY2 = np.array([0.5 + 0.3 * np.sin(angle) for angle in steps])
    
    dataX3 = np.array([0.5 + 0.15 * np.cos(angle) for angle in steps])
    dataY3 = np.array([0.5 + 0.15 * np.sin(angle) for angle in steps])
    
    x = np.concatenate((dataX1, dataX2, dataX3))
    y = np.concatenate((dataY1, dataY2, dataY3))

    samples = np.stack((x, y), axis = 1)
    
    som.train(samples, 100)
    
    labels = [f'{i}' for i in range(1, 4) for k in range(size)]
    som.draw(samples, labels)

def ellipsesGroups():
    som = SOM((20, 15), 2)
    size = 50

    steps = np.linspace(0, 2 * np.pi, size)
    dataX1 = np.array([0.5 + 0.45 * np.cos(angle) for angle in steps])
    dataY1 = np.array([0.5 + 0.25 * np.sin(angle) for angle in steps])
    
    dataX2 = np.array([0.5 + 0.3 * np.cos(angle) for angle in steps])
    dataY2 = np.array([0.5 + 0.125 * np.sin(angle) for angle in steps])
    
    dataX3 = np.array([0.5 + 0.15 * np.cos(angle) for angle in steps])
    dataY3 = np.array([0.5 + 0.075 * np.sin(angle) for angle in steps])
    
    x = np.concatenate((dataX1, dataX2, dataX3))
    y = np.concatenate((dataY1, dataY2, dataY3))

    samples = np.stack((x, y), axis = 1)
    
    som.train(samples, 100)
    
    labels = [f'{i}' for i in range(1, 4) for k in range(size)]
    som.draw(samples, labels)

def noGroups():
    som = SOM((7, 5), 2)
    size = 100
    x = np.random.uniform(0.0, 0.9, size = size)
    y = np.random.uniform(0.0, 0.9, size = size)
   
    samples = np.stack((x, y), axis = 1)

    som.train(samples, 100)

    labels = [f'1' for _ in range(size)]
    som.draw(samples, labels)

def mnistExample():

    trainImages = loadMNIST("train-images.idx3-ubyte")
    trainLabels = loadMNIST("train-labels.idx1-ubyte")
    testImages = loadMNIST("t10k-images.idx3-ubyte")
    testLabels = loadMNIST("t10k-labels.idx1-ubyte")

    numberOfTrainSamples = trainImages[0][0]
    numberOfTestSamples = testImages[0][0]

    # Prepare bare data for analisis.
    resolution = trainImages[0][1] * trainImages[0][2]
    normalizedImages = np.frombuffer(trainImages[1], dtype = np.uint8) / 255.0
    splitedTrainImages = np.reshape(normalizedImages, (numberOfTrainSamples, resolution))

    size = 1000
    images = splitedTrainImages[:size]
    labels = [f'{i}' for i in trainLabels[1][:size]]
    #print(labels)
    som = SOM((10, 10), resolution)

    som.train(images, 100)

    som.draw(images, labels)

uniformGroups()
#normalGroups()
#circleGroups()
#ellipsesGroups()
#noGroups()
#mnistExample()
