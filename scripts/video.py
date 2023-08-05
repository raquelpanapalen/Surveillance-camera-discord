import io
import cv2
import time
import discord
from PIL import Image
import asyncio
from threading import Thread

from scripts.predict import Predictor


class VideoStream:
    def __init__(self, model_path, labels_path, src=0) -> None:
        self.src = src
        self.predictor = Predictor(model_path=model_path, labels_path=labels_path)

        # indicates if the thread is stopped
        self.stopped = True

        # indicates the current person
        self.person = None

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
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            result = self.predictor.predict(img)

            # si se detectaba persona y ahora no, actualizar (pero no enviar mensaje)
            # si no se detectaba persona y ahora si, actualizar y enviar mensaje
            # si se detectaba persona y ahora se detecta la misma, no enviar mensaje
            # si se detectaba persona y ahora se detecta otra, enviar mensaje

            if not result:
                self.person = None
            elif not self.person or result[0] != self.person:
                self.person = result[0]
                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                with io.BytesIO() as output:
                    img.save(output, 'PNG')
                    output.seek(0)
                    await ctx.send(
                        f'Person detected: {result[0]}.',
                        file=discord.File(fp=output, filename='now.png'),
                    )

            time.sleep(60)

    def read(self):
        _, frame = self.stream.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()
        cv2.destroyAllWindows()
