import sys
sys.path.append("../../base")
from common import *

class DummyDataProcessor:
    def __init__(self, config):
        self.name = "dummy_data_processor"
        self.config = config
