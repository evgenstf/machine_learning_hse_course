import sys
sys.path.append("../base")
from common import *
from data_provider import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

with open(sys.argv[1]) as json_data:
    config = json.load(json_data)
    print("config:", config)
    reader = DataProvider(config["data_provider"])
