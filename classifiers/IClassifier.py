#!/usr/bin/python

import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class IClassifier:
    def __init__(self,
                 name,
                 pathToModel,
                 saveModelInFile,
                 classifier):
        self.name = name
        self.pathToModel = pathToModel
        self.saveModelInFile = saveModelInFile
        self.classifier = classifier
        self.model = None

    def __str__(self):
        return self.name

    # Train the classifier with parameter data, or load it from a file.
    # After that we use self.model to do our experimentations,
    # because it contains the trained model.
    def trainClassifier(self, tFeatures, tLabels):
        if self.saveModelInFile is True:
            self.classifier.fit(tFeatures, tLabels)
            self.model = self.classifier
            self.saveModel()
        else:
            self.loadModel()

    # Loads the trained model from a file
    def loadModel(self):
        with open(self.pathToModel, 'rb') as file:
            self.model = pickle.load(file)

    # Saves the trained model in a file
    def saveModel(self):
        if not os.path.exists(os.path.dirname(self.pathToModel)):
            os.makedirs(os.path.dirname(self.pathToModel))
        with open(self.pathToModel, 'wb') as file:
            pickle.dump(self.model, file)

    # Results section. Use self.model beyond this point

    # Get Accuracy as percentage
    def getAccuracy(self, vFeatures, vLabels):
        validationPredicted = self.model.predict(vFeatures)

        accuracy = accuracy_score(vLabels, validationPredicted)
        return accuracy * 100

    def getF1Score(self, vFeatures, vLabels):
        validationPredicted = self.model.predict(vFeatures)
        f1 = f1_score(validationPredicted, vLabels, average='macro')
        return f1 * 100

    def savePredictions(self, features, path):
        validationPredicted = self.model.predict(features)

        with open(path, 'w') as file:
            for i in range(len(validationPredicted)):
                file.write('%d,%d\n' % (i + 1, validationPredicted[i]))

