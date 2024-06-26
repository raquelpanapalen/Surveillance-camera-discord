import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

from .utils import extract_embeddings


class Predictor:
    def __init__(self, model_path, labels_path) -> None:
        self.label_encoder = LabelEncoder()
        self.extractor = FaceNet()
        self.label_encoder.classes_ = pickle.load(open(labels_path, 'rb'))
        self.model = pickle.load(open(model_path, 'rb'))

    def predict(self, image):
        # feature extraction
        output = extract_embeddings(extractor=self.extractor, image=image, face=True)
        if not output:
            return None

        samples = np.expand_dims(output['embeddings'], axis=0)

        # feature classification
        predicted_class = self.model.predict(samples)
        predicted_prob = self.model.predict_proba(samples)
        predicted_label = self.label_encoder.inverse_transform(predicted_class)

        result_dict = dict(zip(self.label_encoder.classes_, predicted_prob[0]))
        return (predicted_label[0], result_dict)
