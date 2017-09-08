import dataPrep

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

training_data = list(
    dataPrep.read(dataset='training', path='C:/Users/nnobel/Vrijdagmiddagproject_Jesper_Myrthe_Nienke/Datasets'))

testing_data = list(
    dataPrep.read(dataset='testing', path='C:/Users/nnobel/Vrijdagmiddagproject_Jesper_Myrthe_Nienke/Datasets'))

training_answers, training_images = dataPrep.returnDataToUse(training_data)

testing_answers, testing_images = dataPrep.returnDataToUse(testing_data)