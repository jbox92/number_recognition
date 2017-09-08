import matplotlib as mpl
from matplotlib import pyplot
import numpy as np

def showMNISTFormat(image):
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.gray)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def showNumpyFormat(image):
    MNISTFormatImage = np.zeros(shape=(28, 28))
    for i in range(28):
        for j in range(28):
            MNISTFormatImage[i][j] = image[28*i+j]
    showMNISTFormat(MNISTFormatImage)