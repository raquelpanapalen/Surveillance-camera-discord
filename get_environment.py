import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
COMMAND_PREFIX = os.getenv('COMMAND_PREFIX')
VIDEO_SRC = os.getenv('VIDEO_SRC', 0)
PREDICTION_TIME = int(os.getenv('PREDICTION_TIME', 10))
