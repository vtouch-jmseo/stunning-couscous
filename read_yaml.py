import yaml


def read_yaml(file_path):
    with open(file_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


if __name__ == "__main__":
    conf = read_yaml("./record_conf.yaml")
    print(conf)