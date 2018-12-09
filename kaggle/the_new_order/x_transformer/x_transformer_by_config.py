import sys
sys.path.append("../../base")
from common import *

from dummy_x_transformer import *
from super_x_transformer import *
from catboost_x_transformer import *


#----------x_transformer_by_config----------

def x_transformer_by_config(config):
    x_transormer_config = config["x_transformer"]
    name = x_transormer_config["name"]
    if (name == "dummy"):
        return DummyXTransformer(x_transormer_config)
    if (name == "super"):
        return SuperXTransformer(x_transormer_config)
    if (name == "catboost"):
        return CatboostXTransformer(x_transormer_config)
    logging.fatal("unknown x transformer name: {0}".format(name))
