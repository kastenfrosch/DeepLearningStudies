import json


config_path = ".\\config.json"


class Config:
    def __init__(self, config_json):
        super(Config, self).__init__()

        with open(config_json, "r") as file:
            config = json.load(file)
        self.epochs = config["epochs"]
        self.no_runs = config["no_runs"]
        self.batch_size = config["batch_size"]
        self.model_dir = config["model_dir"]


config = Config(config_path)


