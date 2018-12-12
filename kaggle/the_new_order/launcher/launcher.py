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

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))

data_provider = DataProvider(config["data_provider"])

primary_x_transformer = x_transformer_by_config(config["primary_x_transformer"])
primary_x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)

x_train_transformed = primary_x_transformer.transform(data_provider.x_train)
del data_provider.x_train
x_test_transformed = primary_x_transformer.transform(data_provider.x_test)
del data_provider.x_test
x_to_predict_transformed = primary_x_transformer.transform(data_provider.x_to_predict)
del data_provider.x_to_predict

del primary_x_transformer


secondary_x_transformer = x_transformer_by_config(config["secondary_x_transformer"])
secondary_x_transformer.load_train_data(x_train_transformed, data_provider.y_train)

x_train_transformed = secondary_x_transformer.transform(x_train_transformed)
x_test_transformed = secondary_x_transformer.transform(x_test_transformed)
x_to_predict_transformed = secondary_x_transformer.transform(x_to_predict_transformed)

del secondary_x_transformer

model = model_by_config(config)
model.load_train_data(
        x_train_transformed,
        data_provider.y_train
)

del x_train_transformed
del data_provider.y_train

prediction = model.predict(x_test_transformed)

#print("prediction:", prediction)
score = spearmanr(prediction, data_provider.y_test)[0]
print("score:", score)
score_file = open("%s" % score, 'w')
score_file.write("%s" % score)


if (config["predict_answer"]):
    prediction = model.predict(x_to_predict_transformed)
    log.info("flush result to {0}".format(config["answer_file"]))
    answer_file = open(config["answer_file"], 'w')
    answer_file.write("Id,Label\n")

    for i in range(len(prediction)):
        answer_file.write("%s,%s\n" % (i + 1, int(prediction[i])))
