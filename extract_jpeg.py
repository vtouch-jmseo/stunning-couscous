import os
import cv2

import numpy as np
from multiprocessing import Pool

from util import read_yaml, normalize
from raw import ADIFileReader

NORM_VALUE = 3000


def get_file_paths(root):
    result = list()
    paths = os.listdir(root)
    for path in paths:
        if os.path.splitext(path)[-1] == ".adi":
            result.append(os.path.join(root, path))

    return result


def run(file_path):
    reader = ADIFileReader(file_path)
    shape = reader.get_shape()
    file_path = file_path.split("/")[-1]
    f_name = os.path.splitext(file_path)[0]
    f_path = os.path.join(SAVE_ROOT, f_name)

    if not os.path.isdir(f_path):
        os.mkdir(f_path) 

    i = 0
    for _ in range(shape[-1]):
        infra, _ = reader.get_frame()
        infra_img = normalize(infra, NORM_VALUE)
        cv2.imwrite(os.path.join(f_path, "%06d_infra.jpg" % i), infra_img)
        i += 1

    reader.close()
    

def on_change(pos):
    pass


if __name__ == "__main__":

    conf = read_yaml("./unpack_conf.yaml")
    paths = get_file_paths(conf['binary_root'])

    if not os.path.isdir(conf['jpeg_save_root']):
        os.mkdir(conf['jpeg_save_root'])
    global SAVE_ROOT
    SAVE_ROOT = conf['jpeg_save_root']

    test_path = paths[0]
    # f, shape = read_header(test_path)
    # infra_buffer, depth_buffer = read_binary(f, shape)
    reader = ADIFileReader(test_path)
    infra_buffer, depth_buffer = reader.get_frame()
    reader.close()

    cv2.namedWindow("Infra Window", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("threshold", "Infra Window", 0, 65000, on_change)
    cv2.setTrackbarPos("threshold", "Infra Window", NORM_VALUE)

    save_flag = True
    while True:
        key = cv2.waitKey(1)

        if key == 27:
            save_flag = False
            break
        if key == ord('s'):
            print("save_start")
            break

        NORM_VALUE = cv2.getTrackbarPos("threshold", "Infra Window")
        copied_buffer = infra_buffer.copy()
        test_image = normalize(copied_buffer, NORM_VALUE)
        cv2.imshow("Infra Window", test_image)

    cv2.destroyAllWindows()

    if save_flag:
        n_process = 4 if len(paths) > 4 else len(paths)
        pool = Pool(processes=n_process)
        pool.map(run, paths)
        pool.close()
        pool.join()


