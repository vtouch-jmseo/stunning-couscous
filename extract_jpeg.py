import os
import cv2
import struct
import time

import numpy as np
from multiprocessing import Pool, Process

from read_yaml import read_yaml

NORM_VALUE = 3000


def read_header(file):
    f = open(file, 'rb')
    shape = [struct.unpack('I', f.read(struct.calcsize('I')))[0] for _ in range(3)]

    return f, shape


def read_binary(f, shape):
    infra = [struct.unpack('H', f.read(struct.calcsize('H')))[0] for _ in range(shape[0] * shape[1])]
    depth = [struct.unpack('H', f.read(struct.calcsize('H')))[0] for _ in range(shape[0] * shape[1])]
    infra = np.array(infra).reshape(shape[1], shape[0])
    depth = np.array(depth).reshape(shape[1], shape[0])
    
    return infra, depth


def normalize(mat, threshold):
    mat[mat>threshold] = threshold
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    mat = np.asarray(mat, dtype=np.uint8)

    return mat


def get_file_paths(root):
    result = list()
    paths = os.listdir(root)
    for path in paths:
        if os.path.splitext(path)[-1] == ".adi":
            result.append(os.path.join(root,path))

    return result


def run(file_path):
    f, shape = read_header(file_path)
    print(shape)
    file_path = file_path.split("/")[-1]
    f_name = os.path.splitext(file_path)[0]
    f_path = os.path.join(SAVE_ROOT, f_name)

    if not os.path.isdir(f_path):
        os.mkdir(f_path) 

    i = 0
    for _ in range(shape[-1]):
        infra, _ = read_binary(f, shape)
        infra_img = normalize(infra, NORM_VALUE)
        cv2.imwrite(os.path.join(f_path, "%06d_infra.jpg" % i), infra_img)
        i += 1

    f.close()
    

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
    f, shape = read_header(test_path)
    infra_buffer, depth_buffer = read_binary(f, shape)
    f.close()
    # infra_buffer, _ = load_binary_file(test_path)
    # test_buffer = infra_buffer[0]
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


