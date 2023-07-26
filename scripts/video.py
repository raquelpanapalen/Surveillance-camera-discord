import cv2
from PIL import Image
import asyncio
from threading import Thread

from scripts.predict import Predictor


class VideoStream:
    def __init__(self, model_path, labels_path, src=0) -> None:
        self.src = src
        # self.predictor = Predictor(model_path=model_path, labels_path=labels_path)

        # indicates if the thread is stopped
        self.stopped = True

    def start(self, ctx):
        # start the thread to read frames
        self.stream = cv2.VideoCapture(self.src)
        self.stopped = False
        self.thread = Thread(target=self.handleThread, args=(ctx,))
        self.thread.start()
        return self

    def handleThread(self, ctx):
        asyncio.run(self.predict(ctx))

    async def predict(self, ctx):
        # keep looping infinitely until thread is stopped
        while True:
            if self.stopped:
                return

            # otherwise, read next frame
            _, self.frame = self.stream.read()

    def read(self):
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()
        cv2.destroyAllWindows()
