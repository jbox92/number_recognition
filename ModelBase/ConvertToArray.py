from PIL import Image
import numpy as np

def convertToArray(filename):
    im = Image.open(filename)
    image = im.load()
    imageArray=np.zeros(28*28)
    for i in range(28):
        for j in range(28):
            imageArray[i*28+j] = image[j,i]
    return imageArray