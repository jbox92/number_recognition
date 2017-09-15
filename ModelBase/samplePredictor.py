import showImage
from sklearn.externals import joblib

def predictSample(filename, image, label):
    clf = joblib.load(filename)
    predictA = label
    predictQ = image

    print("Prediction:", clf.predict([predictQ]))
    print("Actual answer", predictA, "\n")

    showImage.showNumpyFormat(image)