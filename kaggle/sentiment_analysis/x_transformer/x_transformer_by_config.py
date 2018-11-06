import sys
sys.path.append("../../base")
from common import *

from dummy_x_transformer import *
from word_counter_x_transformer import *
from tfidf_x_transformer import *

def x_transformer_by_config(config, need_stemmer):
    x_transormer_config = config["x_transformer"]
    name = x_transormer_config["name"]
    if (name == "dummy"):
        return DummyXTransformer(x_transormer_config)
    if (name == "word_counter"):
        return WordCounterXTransformer(x_transormer_config)
    if (name == "tfidf"):
        return TfidfXTransformer(x_transormer_config, need_stemmer)
    logging.fatal("unknown x transformer name: {0}".format(name))
