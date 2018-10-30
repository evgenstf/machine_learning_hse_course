import sys
sys.path.append("../base")
from common import *
from data_processor_by_config import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

with open(sys.argv[1]) as json_data:
    data_processor = data_processor_by_config(json.load(json_data))
    print("inited data_processor:", data_processor.name)
