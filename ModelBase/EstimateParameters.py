import numpy as np
from sklearn.model_selection import *
from sklearn.svm import SVC
# CROSSVALIDATION

def crossvalidation(x_train, y_train, x_test, y_test):
    svm = SVC()

    cv_performance = cross_val_score(svm, x_train, y_train,
    cv=10)

    test_performance = svm.fit(x_train, y_train).score(x_test,
    y_test)

    print ('Cross-validation accuracy score: %0.3f,'
    ' test accuracy score: %0.3f'
    % (np.mean(cv_performance),test_performance))


# ESTIMATE BEST PARAMETERS
def estimateParameters(x_train,y_train, x_test, y_test):
    learning_algo = SVC(kernel='linear', random_state=101)

    search_space = [{'kernel': ['linear'],
    'C': np.logspace(-3, 3, 7),
    'gamma': np.logspace(-3, 2, 6)},
    {'kernel': ['rbf'],
    'C':np.logspace(-3, 3, 7),
    'gamma': np.logspace(-3, 2, 6)}]

    gridsearch = GridSearchCV(learning_algo,
    param_grid=search_space,
    refit=True, cv=10)

    gridsearch.fit(x_train,y_train)

    print ('Best parameter: %s'
    % str(gridsearch.best_params_))
    cv_performance = gridsearch.best_score_
    test_performance = gridsearch.score(x_test, y_test)
    print ('Cross-validation accuracy score: %0.3f,'
    ' test accuracy score: %0.3f'
    % (cv_performance,test_performance))