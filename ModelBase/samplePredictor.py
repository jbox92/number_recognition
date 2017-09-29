from showImage import showNumpyFormat
from sklearn.externals import joblib
from math import isclose

def predictSample(filename, image, label):
    clf = joblib.load(filename)

    prediction = clf.predict([image])

    print("Prediction:", prediction)
    print("Actual answer", label, "\n")
    print("Correct" if isclose(prediction, label) else "Wrong")

    #showNumpyFormat(image)
