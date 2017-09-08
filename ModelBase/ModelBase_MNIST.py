from sklearn import svm
import Training as train
import Testing as test

# Import of support vector machine (svm)

from sklearn.neural_network import MLPClassifier

"""--------------------------------SETTINGS---------------------------------"""

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

"""------------------------------TRAINING------------------------------------"""
clf = svm.SVC(C=0.001, kernel='linear')

print("Training...")
train.training(training_images, training_answers, 'lerenLezen.pkl', clf)

"""------------------------------Testing Function----------------------------"""
print("Testing...")
test.testing(testing_images, testing_answers, 'lerenLezen.pkl')
