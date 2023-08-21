import os
import cv2
import argparse
from PIL import Image

from get_environment import VIDEO_SRC


class CameraFrames:
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.stream = cv2.VideoCapture(VIDEO_SRC)

    def save_img(self, frame, person_name):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # check if dir exists
        if not os.path.exists(f'{self.data_path}/{person_name}'):
            os.makedirs(f'{self.data_path}/{person_name}')

        num = len(os.listdir(f'{self.data_path}/{person_name}')) + 1
        img.save(f'{self.data_path}/{person_name}/{num}.jpg')

    def run(self):
        while True:
            _, frame = self.stream.read()
            if frame is not None:
                cv2.imshow('frame', frame)
                # Press s to save the frame
                if cv2.waitKey(22) & 0xFF == ord('s'):
                    person_name = input('Introduce tu nombre: ')
                    self.save_img(frame, person_name)
                    print('*' * 20)
                # Press q to close the video windows before it ends if you want
                elif cv2.waitKey(22) & 0xFF == ord('q'):
                    break

        # When everything done, release the capture
        self.stream.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data',
        dest='data_path',
        default=f'{os.getcwd()}/camera_data',
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    camera = CameraFrames(data_path=args.data_path)
    camera.run()
