import sys
sys.path.append("../base")
sys.path.append("../data_provider")
sys.path.append("../x_transformer")

from common import *
from data_provider import *
from x_transformer_by_config import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()


log = logging.getLogger("Launcher")

config_file = open(sys.argv[1], 'r')
config = json.load(config_file)
log.info("launcher config: {0}".format(config))
data_provider = DataProvider(config["data_provider"])

x_transformer = x_transformer_by_config(config)

x_transformer.load_train(data_provider.x_train, data_provider.y_train)
print(x_transformer.word_weights)
