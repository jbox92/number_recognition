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
toTest = input("Test performance? (y/n): ") == "y"
print("Initializing...")


dir = os.path.dirname(__file__)
pathToDataset = os.path.join(dir, 'Datasets')

if toTrain:
    training_data = list(
        dataPrep.read(dataset='training', path=pathToDataset))
    training_data = training_data[0:60000:5]

if toTest:
    testing_data = list(
        dataPrep.read(dataset='testing', path=pathToDataset))

print("Reshaping Data...")
if toTrain:
    training_answers, training_images = dataPrep.returnDataToUse(training_data)
if toTest:
    testing_answers, testing_images = dataPrep.returnDataToUse(testing_data)

import EstimateParameters as est
import ConvertToArray

testImages = ["0_1.bmp", "0_2.bmp", "0_3.bmp", "1_1.bmp", "1_2.bmp", "1_3.bmp","2_1.bmp", "2_2.bmp", "2_3.bmp","3_1.bmp", "3_2.bmp", "3_3.bmp", "4_1.bmp", "4_2.bmp", "4_3.bmp","5_1.bmp", "5_2.bmp", "5_3.bmp","6_1.bmp", "6_2.bmp", "6_3.bmp","7_1.bmp", "7_2.bmp", "7_3.bmp","8_1.bmp", "8_2.bmp", "8_3.bmp","9_1.bmp", "9_2.bmp", "9_3.bmp"]
testingarray = np.zeros((500, 28*28))
answerarray = np.zeros(500)
answers = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
count = 0
for i in range(len(answerarray)):
    if i % 3 == 0:
        count += 1
    if i % 30:
        count = 0
    answerarray[i] = count

j = 0
for i in range(len(testImages)):
    if i % 30:
        j = 0
    testingarray[i] = ConvertToArray.convertToArray(testImages[j])
    j += 1

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
    clf = svm.SVC(C=100, kernel='linear', gamma = 0.001)
    train.training(training_images, training_answers, 'latestNN.pkl', clf)

"""------------------------------Testing Function----------------------------"""

if toTest:
    print("Testing...")
    test.testing(testing_images, testing_answers, 'latestNN.pkl')

"""------------------------------Estimate parameters----------------------------"""

#est.estimateParameters(training_images, training_answers, testing_images, testing_answers)

"""------------------------------ Guess a number ----------------------------"""
for i in range(len(testImages)):
    ourImage = ConvertToArray.convertToArray(testImages[i])
    samplePredictor.predictSample('latestNN.pkl', ourImage, answers[i])