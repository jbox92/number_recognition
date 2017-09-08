import showImage

def predictSample(clf, image, label):
    predictA = label
    predictQ = image

    print("Prediction:", clf.predict([predictQ]))
    print("Actual answer", predictA, "\n")

    showImage.showMNISTFormat(image)