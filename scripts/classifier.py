import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Classifier:
    def __init__(self, data_path, model_path, labels_path, verbose=False) -> None:
        self.data = pickle.load(open(data_path, 'rb'))
        self.model_path = model_path
        self.labels_path = labels_path
        self.label_encoder = LabelEncoder()
        self.normalizer = Normalizer(norm='l2')
        self.model = SVC(kernel='linear', probability=True)
        self.verbose = verbose

    def fit_classifier(self):
        # normalize input data
        train_emb = self.normalizer.transform(self.data['train_emb'])
        test_emb = self.normalizer.transform(self.data['test_emb'])

        # encode labels
        self.label_encoder.fit(self.data['train_y'])
        train_y = self.label_encoder.transform(self.data['train_y'])
        test_y = self.label_encoder.transform(self.data['test_y'])

        # fit svm classifier model
        self.model.fit(train_emb, train_y)

        # Save model
        pickle.dump(self.model, open(self.model_path, 'wb'))
        pickle.dump(self.label_encoder.classes_, open(self.labels_path, 'wb'))

        # predict
        predict_train = self.model.predict(train_emb)
        predict_test = self.model.predict(test_emb)
        predict_test_names = self.label_encoder.inverse_transform(predict_test)

        # get the confidence score
        probabilities = self.model.predict_proba(test_emb)
        confidence = np.max(probabilities, axis=1)

        # Accuracy
        acc_train = accuracy_score(train_y, predict_train)
        acc_test = accuracy_score(test_y, predict_test)

        print(f'Train accuracy: {acc_train} \nTest accuracy: {acc_test}')
        if self.verbose:
            for i, (pred_label_name, score) in enumerate(
                zip(predict_test_names, confidence)
            ):
                print(
                    f'[TEST {i}] Real label: {self.data["test_y"][i]}'
                    f'\t Predicted label: {pred_label_name}'
                    f'\t Confidence: {score}'
                )

            cm = confusion_matrix(
                self.data['test_y'],
                predict_test_names,
                labels=self.label_encoder.classes_,
            )
            print(cm)
