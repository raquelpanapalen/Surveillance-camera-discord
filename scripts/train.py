import os
import argparse

from create_dataset import Dataset
from classifier import Classifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        dest='model_path',
        default=f'{os.getcwd()}/model.pkl',
    )
    parser.add_argument(
        '-l',
        '--labels',
        dest='labels_path',
        default=f'{os.getcwd()}/labels.npy',
    )
    parser.add_argument(
        '-d',
        '--data',
        dest='data_path',
        default=f'{os.getcwd()}/data',
    )
    parser.add_argument(
        '-v', '--verbose', dest='verbose', default=False, action='store_true'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    dataset = Dataset(data_path=args.data_path, verbose=args.verbose)
    dataset.save_data()
    classifier = Classifier(
        data_path=f'{args.data_path}/data.npy',
        model_path=args.model_path,
        labels_path=args.labels_path,
        verbose=args.verbose,
    )
    classifier.fit_classifier()
