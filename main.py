#!/usr/bin/python

import argparse
import os
import warnings
from classifiers.DecisionTreeClassifier import DecisionTreeClassifier
from classifiers.MultiLayerPerceptronClassifier import MultiLayerPerceptronClassifier
from classifiers.NaiveBayesClassifier import BernoulliNaiveBayesClassifier
from classifiers.NaiveBayesClassifier import ComplementNaiveBayesClassifier
from classifiers.NaiveBayesClassifier import GaussianNaiveBayesClassifier

# from classifiers.MultiLayerPerceptronClassifier import MultiLayerPerceptronBestParamsClassifier


def extractDataFromDataset(pathToDataset):
    with open(pathToDataset, "r") as file:
        data = [line.split(',') for line in file.read().split('\n')[:-1]]
    data = [[int(element) for element in row] for row in data]
    features = [d[:-1] for d in data]
    labels = [d[-1] for d in data]

    return features, labels


def extractFeaturesFromDataset(pathToDataset):
    with open(pathToDataset, "r") as file:
        data = [line.split(',') for line in file.read().split('\n')[:-1]]
    features = [[int(element) for element in row] for row in data]

    return features


def showResults(classifier, vFeatures, vLabels):
    print("[Validation set] Accuracy: " + str(classifier.getAccuracy(vFeatures, vLabels)))
    print("[Validation set] F-Measure: " + str(classifier.getF1Score(vFeatures, vLabels)))


def savePredictions(classifier, features, path):
    classifier.savePredictions(features, path)
    print("Predictions correctly saved in ", path)


def experimentOnDataset(path, prefix, saveModel):
        tFeatures, tLabels = extractDataFromDataset(path + prefix + 'Train.csv')
        vFeatures, vLabels = extractDataFromDataset(path + prefix + 'Val.csv')
        testFeatures = extractFeaturesFromDataset(path + prefix + 'Test.csv')

        # Naive Bayes
        nBClassifier = BernoulliNaiveBayesClassifier('models/' + prefix + '/nBModel.pkl', saveModel)
        nBClassifier.trainClassifier(tFeatures, tLabels)
        showResults(nBClassifier, vFeatures, vLabels)

        savePredictions(nBClassifier, vFeatures, prefix + 'Val-nb.csv')
        savePredictions(nBClassifier, testFeatures, prefix + 'Test-nb.csv')


        # Decision Tree
        dTClassifier = DecisionTreeClassifier('models/' + prefix + '/tDModel.pkl', saveModel)
        dTClassifier.trainClassifier(tFeatures, tLabels)
        showResults(dTClassifier, vFeatures, vLabels)

        savePredictions(dTClassifier, vFeatures, prefix + 'Val-dt.csv')
        savePredictions(dTClassifier, testFeatures, prefix + 'Test-dt.csv')


        # Multi Layer Perceptron
        mLPClassifier = MultiLayerPerceptronClassifier('models/' + prefix + '/mLPModel.pkl', saveModel)
        mLPClassifier.trainClassifier(tFeatures, tLabels)
        showResults(mLPClassifier, vFeatures, vLabels)

        savePredictions(dTClassifier, vFeatures, prefix + 'Val-3.csv')
        savePredictions(dTClassifier, testFeatures, prefix + 'Test-3.csv')


def main():
    parser = argparse.ArgumentParser(description="Machine Learning experimentations")
    parser.add_argument('-s', action='store_true', help='Train algorithms and save outputed models to models/', )

    saveModels = parser.parse_args().s
    if saveModels is False and len(os.listdir('./models')) == 0:
        print('Models folder is empty. Train models by starting the program with -s.')
        exit(0)

    print("Dataset 1")
    experimentOnDataset('dataset/ds1/', 'ds1', saveModels)
    print("Dataset 2")
    experimentOnDataset('dataset/ds2/', 'ds2', saveModels)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
