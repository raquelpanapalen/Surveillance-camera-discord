import cv2
from threading import Thread

from scripts.predict import Predictor


class VideoStream:
    def __init__(self, src=0) -> None:
        # initialize video camera stream and predictor
        self.stream = cv2.VideoCapture(src)
        # self.predictor = Predictor()

        # indicates if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely until thread is stopped
        while True:
            if self.stopped:
                return

            # otherwise, read next frame
            _, self.frame = self.stream.read()
            print(self.frame)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()
        cv2.destroyAllWindows()
