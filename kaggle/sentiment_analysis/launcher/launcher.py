import sys
sys.path.append("../base")
sys.path.append("../data_provider")
sys.path.append("../x_transformer")
sys.path.append("../model")

from common import *
from data_provider import *
from x_transformer_by_config import *
from model_by_config import *

def print_weights(x_transformer, model):
    f_weights = zip(x_transformer.features(), model.weights())
    f_weights = sorted(f_weights, key=lambda i: i[1])
    for i in range(1,10):
        print('%s, %.2f' % f_weights[-i])

    print('...')
    for i in reversed(range(1,10)):
        print('%s, %.2f' % f_weights[i])

def calculate_header_predictions(config, data_provider):
    x_transformer = x_transformer_by_config(config)
    model = model_by_config(config)

    x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
    model.load_train(x_transformer.transform(data_provider.x_train), data_provider.y_train)

    prediction = model.predict_probabilities(x_transformer.transform(data_provider.x_to_predict))

    print_weights(x_transformer, model)

    return prediction

def calculate_body_predictions(config, data_provider):
    x_transformer = x_transformer_by_config(config)
    model = model_by_config(config)

    x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
    model.load_train(x_transformer.transform(data_provider.x_train), data_provider.y_train)

    prediction = model.predict_probabilities(x_transformer.transform(data_provider.x_to_predict))

    return prediction

def calculate_final_prediction(header_prediction, body_prediction, config, log,
        header_data_provider, body_data_provider, expected):
    final_prediction = []
    header_weight = config["header_weight"]
    min_total_case_probability = config["min_total_case_probability"]
    negative_count = positive_count = unknown_count = 0
    unknown_file = open(config["unknown_file"], 'w')
    for i in range(len(header_prediction)):
        negative_probability = header_prediction[i][0] * header_weight + body_prediction[i][0] * (1.0 - header_weight)
        positive_probability = header_prediction[i][1] * header_weight + body_prediction[i][1] * (1.0 - header_weight)

        if (positive_probability >= min_total_case_probability):
            final_prediction.append(1)
            positive_count += 1
            continue
        if (negative_probability >= min_total_case_probability):
            final_prediction.append(0)
            negative_count += 1
            continue

        unknown_count += 1
        unknown_file.write(
                "header_prediction: {0} body_prediction: {1} expected: {2}\n\"{3}\"    \"{4}\n".format(
                header_prediction[i], body_prediction[i], expected[i],
                header_data_provider.x_to_predict[i],
                body_data_provider.x_to_predict[i]
            )
        )

        if (positive_probability >= negative_probability):
            final_prediction.append(1)
            positive_count += 1
            continue
        else:
            final_prediction.append(0)
            negative_count += 1
            continue
    log.info("predicted results: positive: {0} negative: {1} unknown: {2}".format(
        positive_count, negative_count, unknown_count))
    return final_prediction

def flush_wrong_predictions(expected, final_prediction, config):
    x_to_predict = open(config["x_to_predict"]).readlines()
    wrong_predictions_file = open(config["wrong_predictions_file"], "w")
    for i in range(len(final_prediction)):
        if final_prediction[i] != -1 and final_prediction[i] != expected[i]:
            wrong_predictions_file.write("line: %s expected: %s predicted: %s text: %s\n" % (i + 2,
                expected[i], final_prediction[i], x_to_predict[i]))

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

log = logging.getLogger("Launcher")

config_file = open(sys.argv[1], 'r')
config = json.load(config_file)
log.info("launcher config: {0}".format(config))
#data_provider = DataProvider(config["data_provider"])
header_data_provider = DataProvider(config["header_data_provider"], "header")
body_data_provider = DataProvider(config["body_data_provider"], "body")

header_prediction = calculate_header_predictions(config, header_data_provider)
body_prediction = calculate_body_predictions(config, body_data_provider)

expected = pd.read_csv(config["magic"])["Probability"].tolist()
final_prediction = calculate_final_prediction(header_prediction, body_prediction, config, log,
        header_data_provider, body_data_provider, expected)

print("ratio_score:", ratio_score(expected, final_prediction))
flush_wrong_predictions(expected, final_prediction, config)

answer_file = open(config["answer_file"], "w")
answer_file.write("Id,Probability\n")
for i in range(len(final_prediction)):
    answer_file.write("%s,%s.0\n" % (i + 1, final_prediction[i]))

