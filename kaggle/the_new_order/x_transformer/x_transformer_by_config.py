import sys
sys.path.append("../../base")
from common import *

from dummy_x_transformer import *
from super_x_transformer import *
from catboost_x_transformer import *
from regboost_x_transformer import *

#----------x_transformer_by_config----------

def x_transformer_by_config(config):
    name = config["name"]
    if (name == "dummy"):
        return DummyXTransformer(config)
    if (name == "super"):
        return SuperXTransformer(config)
    if (name == "catboost"):
        return CatboostXTransformer(config)
    if (name == "regboost"):
        return RegboostXTransformer(config)
    logging.fatal("unknown x transformer name: {0}".format(name))
