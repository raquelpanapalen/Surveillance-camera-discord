import numpy as np
from PIL import Image
from keras_facenet import FaceNet


def extract_embeddings(image, face=False, verbose=False):
    output = {}
    embedder = FaceNet()
    results = embedder.extract(image, threshold=0.95)
    print(results)
    if not len(results) or len(results) > 1:
        return None

    output['embeddings'] = results[0]['embedding']

    if face:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = image[y1:y2, x1:x2]
        face_image = Image.fromarray(face).tobytes()
        output['face'] = face_image

    if verbose:
        print(f'Face detected with confidence: {results[0]["confidence"]}')

    return output
