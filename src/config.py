import yaml

def get_config(config: str):
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)