from pickletools import uint8

import dataPrep
from sklearn import svm
import Training as train
import Testing as test
import numpy as np
import samplePredictor
import os


"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
toTrain = input("Pretrained model? (y/n): ") == "n"

print("Initializing...")


dir = os.path.dirname(__file__)
pathToDataset = os.path.join(dir, 'Datasets')

if toTrain:
    training_data = list(
        dataPrep.read(dataset='training', path=pathToDataset))
    training_data = training_data[0:60000:100]

testing_data = list(
    dataPrep.read(dataset='testing', path=pathToDataset))

print("Reshaping Data...")
if toTrain:
    training_answers, training_images = dataPrep.returnDataToUse(training_data)
    print(training_images[0])
testing_answers, testing_images = dataPrep.returnDataToUse(testing_data)


# Import of support vector machine (svm)

from sklearn.neural_network import MLPClassifier

"""--------------------------------SETTINGS---------------------------------"""

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

"""------------------------------TRAINING------------------------------------"""
if toTrain :
    print("Training...")
    clf = svm.SVC(C=0.001, kernel='linear')
    train.training(training_images, training_answers, 'latestNN.pkl', clf)

"""------------------------------Testing Function----------------------------"""
toTest = input("Test performance? (y/n): ") == "y"
if toTest:
    print("Testing...")
    test.testing(testing_images, testing_answers, 'latestNN.pkl')


"""------------------------------ Guess a number ----------------------------"""
import ConvertToArray

ourImage = ConvertToArray.convertToArray('test.bmp')
samplePredictor.predictSample('trainedWithWholeMNIST.pkl', ourImage, 4)