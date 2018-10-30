import sys
sys.path.append("../base")
from common import *

if (len(sys.argv) < 1):
    print("Usage: ./example.py <config>")
    exit()

config = json.loads()
