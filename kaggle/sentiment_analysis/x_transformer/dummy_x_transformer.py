import sys
sys.path.append("../../base")
from common import *

class DummyXTransformer:
    def __init__(self, config):
        self.name = "dummy"
        self.config = config
