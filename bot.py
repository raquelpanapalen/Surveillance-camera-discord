import io
import os
import argparse
import discord
import asyncio
from discord import Embed
from discord.ext import commands

from scripts.video import VideoStream
from get_environment import DISCORD_TOKEN, COMMAND_PREFIX, VIDEO_SRC


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
        default=f'{os.getcwd()}/labels.pkl',
    )
    args = parser.parse_args()
    return args


class DiscordBot:
    def __init__(self, model_path, labels_path) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = commands.Bot(COMMAND_PREFIX, intents=intents)
        self.token = DISCORD_TOKEN
        self.vs = VideoStream(
            model_path=model_path, labels_path=labels_path, src=VIDEO_SRC
        )

        @self.client.event
        async def on_ready():
            self.loop = asyncio.get_running_loop()
            print('Logged in as {}'.format(self.client.user.name))

        @self.client.command()
        async def info(ctx):
            description = (
                """
                **"""
                + COMMAND_PREFIX
                + """ activate** -> Activate surveillance camera (face detection & recognition)
                **"""
                + COMMAND_PREFIX
                + """ deactivate** -> Deactivate surveillance camera
                    **"""
                + COMMAND_PREFIX
                + """ now** -> Get current camera image
                """
            )

            embed = Embed(title="Bot commands", description=description)
            await ctx.send(embed=embed)

        @self.client.command()
        async def activate(ctx):
            await ctx.send('Surveillance camera is active.')
            self.vs.start(ctx, self.loop)

        @self.client.command()
        async def deactivate(ctx):
            if self.vs.stopped:
                await ctx.send(
                    'The surveillance camera is not active. Activate it first with /activate.'
                )
                return
            await ctx.send('Surveillance camera has been deactivated.')
            self.vs.stop()

        @self.client.command()
        async def now(ctx):
            if self.vs.stopped:
                await ctx.send(
                    'The surveillance camera is not active. Activate it first with /activate.'
                )
                return
            frame = self.vs.read()
            with io.BytesIO() as output:
                frame.save(output, 'PNG')
                output.seek(0)
                await ctx.send(file=discord.File(fp=output, filename='now.png'))

    def start(self):
        self.client.run(self.token)


if __name__ == "__main__":
    args = get_args()
    bot = DiscordBot(model_path=args.model_path, labels_path=args.labels_path)
    bot.start()
