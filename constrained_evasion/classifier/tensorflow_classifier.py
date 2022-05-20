import numpy as np


class TensorflowClassifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, x, y):
        self.classifier.fit(x, y)

    def predict_proba(self, x):
        return self.classifier.predict(x)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def get_internal_classifier(self):
        return self.classifier
