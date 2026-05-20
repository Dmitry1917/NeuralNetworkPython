import numpy as np

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

