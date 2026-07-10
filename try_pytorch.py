import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

import numpy as np
import matplotlib.pyplot as plt

from mnist import loadMNIST
from mnist import vectorized

trainImages = loadMNIST("train-images.idx3-ubyte")
trainLabels = loadMNIST("train-labels.idx1-ubyte")
testImages = loadMNIST("t10k-images.idx3-ubyte")
testLabels = loadMNIST("t10k-labels.idx1-ubyte")

numberOfTrainImages = trainImages[0][0]
numberOfTestImages = testImages[0][0]

class CustomMNISTDataset(Dataset):
    def __init__(self, loadedImagesData, loadedLabelsData):
        numberOfSamples = loadedImagesData[0][0]
        resolution = loadedImagesData[0][1] * loadedImagesData[0][2]
        images = np.frombuffer(loadedImagesData[1], dtype = np.uint8).astype(np.float32) / 255.0

        self.numberOfSamples = numberOfSamples
        self.images = torch.from_numpy(np.reshape(images, (numberOfSamples, 1, loadedImagesData[0][1], loadedImagesData[0][2])))
        self.labels = np.frombuffer(loadedLabelsData[1], dtype = np.uint8)

    def __len__(self):
        return self.numberOfSamples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        return (image, label)

trainingData = CustomMNISTDataset(trainImages, trainLabels)
testData = CustomMNISTDataset(testImages, testLabels)

# Standart MNIST version - can be used by uncommenting. It is MUCH slower than handmade version, because it repeating data transformation every epoch.
#trainingData = datasets.MNIST(
#        root = 'data',
#        train = True,
#        download = True,
#        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
#        )
#testData = datasets.MNIST(
#        root = 'data',
#        train = False,
#        download = True,
#        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
#        )
#print(testData)

batch_size = 10

trainDataLoader = DataLoader(trainingData, batch_size = batch_size, shuffle = True)
testDataLoader = DataLoader(testData, batch_size = batch_size, shuffle = True)

labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

# Simple checks of the data.

#for x, y in testDataLoader:
#    print(x.shape)
#    print(y)
#    break
#
#batchFeatures, batchLabels = next(iter(trainDataLoader))
#print(batchFeatures.size(), batchLabels.size())
#img = batchFeatures[0].squeeze()
#label = batchLabels[0]
#print(label)
#plt.imshow(img, cmap = 'gray')
#plt.show()

#figure = plt.figure(figsize = (8, 8))
#cols, rows = 3, 3
#for i in range(1, cols * rows + 1):
#    sampleId = torch.randint(len(trainingData), size = (1,)).item()
#    print(sampleId)
#    img, label = trainingData[sampleId]
#    figure.add_subplot(rows, cols, i)
#    plt.title(labels_map[label])
#    plt.axis('off')
#    plt.imshow(img.squeeze(), cmap = 'gray')
#    #print(img, img.squeeze(), label)
#plt.show()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Use device {device}')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
                nn.Linear(28 * 28, 30),
                nn.Sigmoid(),
                #nn.Dropout(0.1),# In such simple network adding dropout made results worse in all tests.
                nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

for name, param in model.named_parameters():
    print(name)
    print(param.size())
#    print(param[0])

# Usage of softmax in layers above is unnecessary, because of CrossEntropyLoss below - it already contain softmax under loglikehood, like it was used in network.py.

#randInput = torch.rand(1, 28, 28, device = device)
#randRes = model(randInput)
#softMax = nn.Softmax(dim = 1)(randRes)
#maxIndex = randRes.argmax(1)
#maxIndexSoftMax = softMax.argmax(1)
#
##print(randInput)
#print(randRes)
#print(softMax)
#print(maxIndex)
#print(maxIndexSoftMax)

lossFunc = nn.CrossEntropyLoss()
l1R = 0.00005
optimizer = torch.optim.SGD(model.parameters(), lr = 1.0, momentum = 0.9, dampening = 0.9, weight_decay = 0.00001)
lrScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)

def train(dataLoader, model, loss_fn, optimizer):
    size = len(dataLoader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataLoader):
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = loss_fn(prediction, y)

        if l1R > 0:
            l1 = torch.tensor(0.0).to(device)
            for param in model.parameters():
                if len(param.size()) > 1:
                    l1 += param.abs().sum()
            loss += l1 * l1R

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % (100 * 128 / batch_size) == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f'loss: {loss:.4f}, lr: {lrScheduler.get_last_lr()[0]:.5f}, current: {current:05d}, size: {size:05d}')
    lrScheduler.step()

def test(dataLoader, model, loss_fn):
    size = len(dataLoader.dataset)
    numBatches = len(dataLoader)
    model.eval()
    testLoss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            testLoss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    testLoss /= numBatches
    correct /= size
    print(f'Test error:\ncorrect: {100*correct:.4f}%, average loss: {testLoss:.4f}')

epochs = 30
for e in range(epochs):
    print(f'Epoch {e}:')
    train(trainDataLoader, model, lossFunc, optimizer)
    test(testDataLoader, model, lossFunc)

torch.save(model.state_dict(), 'model.pth')# Optimizer and scheduler can be saved too, if needed.

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth', weights_only = True))

model.eval()
classes = list(labels_map)
with torch.no_grad():
    for i in range(10):
        sampleId = torch.randint(len(testData), size = (1,)).item()
        x, y = testData[sampleId][0], testData[sampleId][1]
        x = x.to(device)
        prediction = model(x)
        pred, real = classes[prediction[0].argmax(0)], classes[y]
        print(f'Predicted: {pred}, real: {real}')

#optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
#for e in range(epochs):
#    print(f'Epoch {e}:')
#    train(trainDataLoader, model, lossFunc, optimizer)
#    test(testDataLoader, model, lossFunc)


