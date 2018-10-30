from dummy_data_processor import *

def data_processor_by_config(config):
    data_processor_config = config["data_processor"]
    name = data_processor_config["name"]
    if (name == "dummy"):
        return DummyDataProcessor(data_processor_config)
    logging.fatal("unknown data processor name: {0}".format(name))
