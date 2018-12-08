import sys
sys.path.append("../../base")
from common import *

from dummy_model import *
from catboost_model import *




#----------model_by_config----------

def model_by_config(config):
    model_config = config["model"]
    name = model_config["name"]
    if (name == "dummy"):
        return DummyModel(model_config)
    if (name == "catboost"):
        return CatboostModel(model_config)
    logging.fatal("unknown model name: {0}".format(name))
