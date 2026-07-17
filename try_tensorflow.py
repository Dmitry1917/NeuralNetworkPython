import tensorflow as tf

from mnist import loadMNIST
import numpy as np

print('\n\nActual logs start.\n')
print('tf version is ', tf.__version__)

print('devices: ', tf.config.list_physical_devices())

trainImages = loadMNIST("train-images.idx3-ubyte")
trainLabels = loadMNIST("train-labels.idx1-ubyte")
testImages = loadMNIST("t10k-images.idx3-ubyte")
testLabels = loadMNIST("t10k-labels.idx1-ubyte")

numberOfTrainSamples = trainImages[0][0]
numberOfTestSamples = testImages[0][0]

images = np.frombuffer(trainImages[1], dtype = np.uint8).astype(np.float32) / 255.0
xTrain = np.reshape(images, (numberOfTrainSamples, trainImages[0][1], trainImages[0][2]))
yTrain = np.frombuffer(trainLabels[1], dtype = np.uint8)

images = np.frombuffer(testImages[1], dtype = np.uint8).astype(np.float32) / 255.0
xTest = np.reshape(images, (numberOfTestSamples, testImages[0][1], testImages[0][2]))
yTest = np.frombuffer(testLabels[1], dtype = np.uint8)


# Embedded mnist version.
#mnist = tf.keras.datasets.mnist
#(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
#xTrain, xTest = xTrain / 255.0, xTest / 255.0

l1R = 0.00005
l2R = 0.00001

regularizer = tf.keras.regularizers.L1L2(l1R, l2R)
model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape = (28, 28)),# Replaced by next 2 lines to get rid of warning.
    tf.keras.layers.Input(shape = (28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        30,
        activation = tf.keras.activations.sigmoid,
        kernel_regularizer = regularizer
    ),
    tf.keras.layers.Dense(10, kernel_regularizer = regularizer)
])

batchSize = 10
initialLearningRate = 1.0

def schedulerFn(epoch):
    return initialLearningRate * (0.9 ** epoch)

schedulerCallback = tf.keras.callbacks.LearningRateScheduler(schedulerFn, verbose = 1)

optimizer = tf.keras.optimizers.SGD(
        learning_rate = initialLearningRate,#scheduler,
        #momentum = 0.9# Momentum in tensorflow work differently, compared to pytorch and also do not use dampening, leading to different results with the same values.
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer = optimizer,
              loss = loss,
              metrics = ['accuracy']
)

dataset = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))
dataset = dataset.shuffle(buffer_size = len(xTrain), reshuffle_each_iteration = True)
dataset = dataset.batch(batchSize)
model.fit(dataset, epochs = 30, callbacks = [schedulerCallback], validation_data = [xTest, yTest])

#model.fit(xTrain, yTrain, batch_size = batchSize, epochs = 5)
#model.evaluate(xTest, yTest)

print('\nSave and check model again.\n')
model.save('tfModelMNIST.keras')

loadedModel = tf.keras.models.load_model('tfModelMNIST.keras')
loadedModel.evaluate(xTest, yTest)

samplesIds = tf.keras.random.randint([5], minval = 0, maxval = len(xTest))
x, y = xTest[samplesIds], yTest[samplesIds]
prediction = loadedModel.predict(x)

print(prediction.argmax(axis = 1), y)
