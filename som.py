import numpy as np

def weightsInit(neurons, weightsPerNeuron):
    return np.random.default_rng().uniform(0.0, 1.0, size = (neurons, weightsPerNeuron))

class SOM:
    def __init__(self, neuronsNumbers, weightsNumbers):
        self.neurons = weightsInit(neuronsNumbers, weightsNumbers)

    def findBMU(self, sample) -> int:
        minDistance = 10
        minIndex = 0
        for index, neuron in enumerate(self.neurons):
            distance = np.linalg.norm(neuron - sample)
            if distance < minDistance:
                minDistance = distance
                minIndex = index
        return minIndex

    @staticmethod
    def neighbourhood(bmuIndex, index):
        if index == bmuIndex:
            return 1.0
        else:
            return 1.0 / (2 ** abs(index - bmuIndex))

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


som = SOM(10, 2)
print('original neurons')
print(som.neurons)

size = 10
dataX1 = np.random.uniform(0.1, 0.2, size = size)
dataY1 = np.random.uniform(0.1, 0.2, size = size)

dataX2 = np.random.uniform(0.5, 0.6, size = size)
dataY2 = np.random.uniform(0.5, 0.6, size = size)

dataX3 = np.random.uniform(0.9, 0.95, size = size)
dataY3 = np.random.uniform(0.9, 0.95, size = size)

x = np.concatenate((dataX1, dataX2, dataX3))
y = np.concatenate((dataY1, dataY2, dataY3))

samples = np.stack((x, y), axis = 1)

print('samples')
print(samples)

som.train(samples, 1000)

print('final neurons')
print(som.neurons)
