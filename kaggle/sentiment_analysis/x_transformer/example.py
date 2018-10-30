import sys
sys.path.append("../base")
from common import *
from x_transformer_by_config import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

with open(sys.argv[1]) as json_data:
    x_transformer = x_transformer_by_config(json.load(json_data))
    print("inited x_transformer:", x_transformer.name)
