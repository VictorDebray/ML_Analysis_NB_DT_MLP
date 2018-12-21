#!/usr/bin/python

from classifiers.IClassifier import IClassifier
from sklearn import naive_bayes


class BernoulliNaiveBayesClassifier(IClassifier):
    def __init__(self, pathToModel, saveModel, alpha=1.0):
        super().__init__('Bernoulli Naive Bayes',
                         pathToModel,
                         saveModel,
                         naive_bayes.BernoulliNB(
                             alpha=alpha
                         ))


class MultinomialNaiveBayesClassifier(IClassifier):
    def __init__(self, pathToModel, saveModel, alpha=1.0):
        super().__init__('Multinomial Naive Bayes',
                         pathToModel,
                         saveModel,
                         naive_bayes.MultinomialNB(
                             alpha=alpha
                         ))


class ComplementNaiveBayesClassifier(IClassifier):
    def __init__(self, pathToModel, saveModel, alpha=1.0):
        super().__init__('Complement Naive Bayes',
                         pathToModel,
                         saveModel,
                         naive_bayes.ComplementNB(
                         ))


class GaussianNaiveBayesClassifier(IClassifier):
    def __init__(self, pathToModel, saveModel, alpha=1.0):
        super().__init__('Gaussian Naive Bayes',
                         pathToModel,
                         saveModel,
                         naive_bayes.GaussianNB(
                         ))
