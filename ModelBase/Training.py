from sklearn.externals import joblib

def training(images, answers, filename, classifier):

    classifier.fit(images, answers)

    joblib.dump(classifier, filename)
