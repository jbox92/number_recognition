from sklearn.externals import joblib

def testing(images, answers, filename):
    clf = joblib.load(filename)
    correct = 0
    wrong = 0

    a, q = answers, images
    for i in range(len(q)):
        p = clf.predict([q[i]])
        if a[i] == p[0]:
            correct += 1
        else:
            wrong += 1
    print("Statistics, correct answers:", correct / (correct + wrong))