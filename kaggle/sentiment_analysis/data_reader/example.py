import sys
sys.path.append("../base")
from common import *
from data_reader import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

with open(sys.argv[1]) as json_data:
    reader = DataReader(json.load(json_data))
    while (reader.has_review()):
        print("{", reader.next_review(), "}")
