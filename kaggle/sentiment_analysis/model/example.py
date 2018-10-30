import sys
sys.path.append("../base")
from common import *
from model_by_config import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

with open(sys.argv[1]) as json_data:
    model = model_by_config(json.load(json_data))
    print("inited model:", model.name)
