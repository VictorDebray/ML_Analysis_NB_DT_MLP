#!/usr/bin/python

from classifiers.IClassifier import IClassifier
from sklearn import tree


class DecisionTreeClassifier(IClassifier):
    def __init__(self, pathToModel, saveModel):
        super().__init__('Decision Tree',
                         pathToModel,
                         saveModel,
                         tree.DecisionTreeClassifier())


