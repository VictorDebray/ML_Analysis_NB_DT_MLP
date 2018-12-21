#!/usr/bin/python

from classifiers.IClassifier import IClassifier
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class MultiLayerPerceptronClassifier(IClassifier):
    def __init__(self, pathToModel, saveModel):
        super().__init__('Multi-layer Perceptron',
                         pathToModel,
                         saveModel,
                         neural_network.MLPClassifier(
                             activation='logistic',
                             solver='adam',
                             hidden_layer_sizes=(75,)
                         ))

#
# class MultiLayerPerceptronBestParamsClassifier():
#     def __init__(self, tFeatures, tLabels):
#         param_grid = [
#             {
#                 'hidden_layer_sizes': [
#                     (1,), (10,), (20,), (30,), (40,), (50,), (75,), (100,)
#                 ]
#             }
#         ]
#
#         X_train, X_test, y_train, y_test = train_test_split(
#             tFeatures, tLabels, test_size=0.5, random_state=0)
#
#         scores = ['precision', 'recall']
#
#         for score in scores:
#             print("# Tuning hyper-parameters for %s" % score)
#             print()
#
#             clf = GridSearchCV(neural_network.MLPClassifier(activation='logistic',
#                                                             solver='adam',
#                                                             hidden_layer_sizes=(50,)),
#                                param_grid,
#                                cv=3,
#                                scoring='%s_macro' % score)
#             clf.fit(X_train, y_train)
#
#             print("Best parameters set found on development set:")
#             print()
#             print(clf.best_params_)
#             print()
#             print("Grid scores on development set:")
#             print()
#             means = clf.cv_results_['mean_test_score']
#             stds = clf.cv_results_['std_test_score']
#             for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#                 print("%0.3f (+/-%0.03f) for %r"
#                       % (mean, std * 2, params))
#             print()
#
#             print("Detailed classification report:")
#             print()
#             print("The model is trained on the full development set.")
#             print("The scores are computed on the full evaluation set.")
#             print()
#             y_true, y_pred = y_test, clf.predict(X_test)
#             print(classification_report(y_true, y_pred))
#             print()
#
