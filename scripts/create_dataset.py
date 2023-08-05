import os
import pickle
import numpy as np
from PIL import Image
from keras_facenet import FaceNet

from utils import extract_embeddings


class Dataset:
    def __init__(self, data_path, verbose):
        self.data_path = data_path
        self.extractor = FaceNet()
        self.verbose = verbose

    def load_images(self, dir):
        embeddings = []
        for filename in os.listdir(dir):
            path = f'{dir}/{filename}'
            if self.verbose:
                print(f'Processing: {path}')
            image = Image.open(path)
            image = image.convert('RGB')
            pixels = np.asarray(image)
            output = extract_embeddings(self.extractor, pixels, verbose=self.verbose)
            if not output:
                raise Exception(f'Something has gone wrong with image {path}')
            embeddings.append(output['embeddings'])

        return embeddings

    def load_subset(self, subset, dir):
        # get embeddings for each person
        for subdir in os.listdir(dir):
            embeddings = self.load_images(f'{dir}/{subdir}')
            labels = [subdir for i in range(len(embeddings))]
            if self.verbose:
                print(
                    "[%s]: There are %d images in the class %s:"
                    % (subset.upper(), len(embeddings), subdir)
                )

            self.data[f'{subset}_emb'].extend(embeddings)
            self.data[f'{subset}_y'].extend(labels)

    def save_data(self):
        self.data = {
            'train_y': [],
            'train_emb': [],
            'test_y': [],
            'test_emb': [],
        }
        subsets = ['train', 'test']
        for subset in subsets:
            self.load_subset(subset=subset, dir=f'{self.data_path}/{subset}')

        self.data = {key: np.stack(self.data[key]) for key in self.data.keys()}
        pickle.dump(self.data, open(f'{self.data_path}/data.pkl', 'wb'))
