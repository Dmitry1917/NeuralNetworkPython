import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def weightsInit(neuronsDimensions, weightsPerNeuron):
    return np.random.default_rng().uniform(0.0, 1.0, size = (math.prod(neuronsDimensions), weightsPerNeuron))

class SOM:
    def __init__(self, neuronsDimensions: [int, ...], weightsNumbers):
        self.neuronsDimensions = neuronsDimensions
        self.neurons = weightsInit(neuronsDimensions, weightsNumbers)

    def findBMU(self, sample) -> int:
        minDistance = 10
        minIndex = 0
        for index, neuron in enumerate(self.neurons):
            distance = np.linalg.norm(neuron - sample)
            if distance < minDistance:
                minDistance = distance
                minIndex = index
        return minIndex

    def neighbourhood(self, bmuIndex, index):
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
                diff = abs(bmuX - x) + abs(bmuY - y)
            return 1.0 / (3 ** diff)

    @staticmethod
    def learningRate(epoch, epochs):
        return 1 - epoch / epochs;

    def train(self, samples, epochs):
        localSamples = samples.copy()
        for epoch in range(epochs):
            np.random.shuffle(localSamples)
            for sample in localSamples:
                bmuIndex = self.findBMU(sample)
                for index, neuron in enumerate(self.neurons):
                    neuron = neuron + self.neighbourhood(bmuIndex, index) * self.learningRate(epoch, epochs) * (sample - neuron)
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

    def draw(self, samples = []):
        rows = self.neuronsDimensions[0]
        if len(self.neuronsDimensions) == 1:
            columns = 1
        else:
            columns = self.neuronsDimensions[1]

        width = 800
        height = width * rows // columns
        img = Image.new('RGB', (width, height), color = 'white')
        draw = ImageDraw.Draw(img)

        margin = 20
        neuronAreaWidth = width / columns
        neuronAreaHeight = height / rows
        for i in range(rows):
            for k in range(columns):
                topLeftX = k * neuronAreaWidth + margin / 2
                topLeftY = i * neuronAreaHeight + margin / 2
                bottomRightX = topLeftX + neuronAreaWidth - margin
                bottomRightY = topLeftY + neuronAreaHeight - margin
                draw.rectangle((topLeftX, topLeftY, bottomRightX, bottomRightY), fill = '#555555')

        for sample in samples:
            bmuIndex = self.findBMU(sample)
            bmuY = bmuIndex // columns
            bmuX = bmuIndex % columns

            topLeftX = bmuX * neuronAreaWidth + margin / 2 + np.random.uniform(15)
            topLeftY = bmuY * neuronAreaHeight + margin / 2 + np.random.uniform(15)
            bottomRightX = topLeftX + 2
            bottomRightY = topLeftY + 2
            draw.rectangle((topLeftX, topLeftY, bottomRightX, bottomRightY), fill = '#999999')

        img.show()

som = SOM((7, 5), 2)
print('original neurons')
som.print()

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

print('samples')
print(samples)

som.plotPure2D()

som.train(samples, 100)

print('final neurons')
som.print()
som.plotPure2D()
som.draw(samples)
