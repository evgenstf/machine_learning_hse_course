import sys
sys.path.append("../../base")
from common import *

class DummyXTransformer:
    def __init__(self, config):
        self.name = "dummy_x_transformer"
        self.config = config
