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


log = logging.getLogger("Launcher")

config_file = open(sys.argv[1], 'r')
config = json.load(config_file)
log.info("launcher config: {0}".format(config))
data_provider = DataProvider(config["data_provider"])

x_transformer = x_transformer_by_config(config)
model = model_by_config(config)

#model.load_train(x_transformer.transform(data_provider.x_train), data_provider.y_train)
print(model.load_train(x_transformer.transform(data_provider.x_train), data_provider.y_train))

exit()

prediction = model.predict(x_transformer.transform(data_provider.x_to_predict))
expected = pd.read_csv(config["magic"])["Probability"].tolist()

print(ratio_score(expected, prediction))


wrong_predictions_file = open(config["wrong_predictions_file"], "w")
for i in range(len(prediction)):
    if prediction[i] != -1 and prediction[i] != expected[i]:
        wrong_predictions_file.write("line: %s expected: %s predicted: %s text: %s\n" % (i + 2,
            expected[i], prediction[i], data_provider.x_to_predict[i]))

answer_file = open(config["answer_file"], "w")
answer_file.write("Id,Probability\n")
for i in range(len(prediction)):
    answer_file.write("%s,%s.0\n" % (i + 1, prediction[i]))
