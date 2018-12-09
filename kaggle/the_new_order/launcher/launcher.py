import sys
sys.path.append("../base")
sys.path.append("../data_provider")
sys.path.append("../x_transformer")
sys.path.append("../model")

from common import *
from data_provider import *
from x_transformer_by_config import *
from model_by_config import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

config_file = open(sys.argv[1], 'r')
config = json.load(config_file)


#----------launcher----------

def create_model(config, data_provider):
    x_transformer = x_transformer_by_config(config)
    model = model_by_config(config)

    x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
    model.load_train_data(x_transformer.transform(data_provider.x_train), data_provider.y_train)

    return model

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))

data_provider = DataProvider(config["data_provider"])

model = create_model(config, data_provider)
#prediction = model.predict(data_provider.x_test)

#print("score:", spearmanr(prediction, data_provider.y_test)[0])

prediction = model.predict(data_provider.x_to_predict)

if (config["flush_to_file"]):
    log.info("flush result to {0}".format(config["answer_file"]))
    answer_file = open(config["answer_file"], 'w')
    answer_file.write("Id,Label\n")

    for i in range(len(prediction)):
        answer_file.write("%s,%s\n" % (i + 1, int(prediction[i][0])))
