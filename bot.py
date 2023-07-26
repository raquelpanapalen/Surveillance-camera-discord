import io
import cv2
from threading import Thread
import discord
from discord.ext import commands

from scripts.video import VideoStream
from get_environment import DISCORD_TOKEN, COMMAND_PREFIX, VIDEO_SRC


class DiscordBot:
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = commands.Bot(COMMAND_PREFIX, intents=intents)
        self.token = DISCORD_TOKEN

        @self.client.event
        async def on_ready():
            print('Logged in as {}'.format(self.client.user.name))

        @self.client.command()
        async def test(ctx, arg):
            await ctx.send(arg)

        @self.client.command()
        async def activate(ctx):
            await ctx.send('Surveillance camera is active.')
            self.vs = VideoStream(VIDEO_SRC)
            self.vs.start()

        @self.client.command()
        async def deactivate(ctx):
            await ctx.send('Surveillance camera has been deactivated.')
            self.vs.stop()

    def start(self):
        self.client.run(self.token)


if __name__ == "__main__":
    bot = DiscordBot()
    bot.start()
