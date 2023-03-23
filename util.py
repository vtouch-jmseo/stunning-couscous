import cv2
import yaml

import numpy as np


def read_yaml(file_path):
    with open(file_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


def normalize(mat, threshold):
    mat[mat > threshold] = threshold
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    mat = np.asarray(mat, dtype=np.uint8)

    return mat


if __name__ == "__main__":
    conf = read_yaml("./record_conf.yaml")
    print(conf)