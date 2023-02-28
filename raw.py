import os
import cv2
import struct

import numpy as np


def load_binary_file(file):
    infra_buffer = list()
    depth_buffer = list()
    with open(file, 'rb') as f:
        shape = [struct.unpack('I', f.read(struct.calcsize('I')))[0] for _ in range(3)]
        print(shape)
        for _ in range(shape[-1]):
            infra = [struct.unpack('H', f.read(struct.calcsize('H')))[0] for _ in range(shape[0] * shape[1])]
            depth = [struct.unpack('H', f.read(struct.calcsize('H')))[0] for _ in range(shape[0] * shape[1])]
            infra = np.array(infra).reshape(shape[1], shape[0])
            depth = np.array(depth).reshape(shape[1], shape[0])
            infra_buffer.append(infra)
            depth_buffer.append(depth)

    return infra_buffer, depth_buffer

def normalize(mat):
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    mat = np.asarray(mat, dtype=np.uint8)

    return mat

if __name__=="__main__":
    file_path = "./20230227_221826.adi"

    infra_buffer, depth_buffer = load_binary_file(file_path)

    i = 0
    for infra, depth in zip(infra_buffer, depth_buffer):
        depth_img = normalize(depth)
        infra_img = normalize(infra)
        infra_img = np.roll(infra_img, -5)  # TODO: removeS
        cv2.imwrite("./%03d_depth.jpg" % i, depth_img)
        cv2.imwrite("./%03d_infra.jpg" % i, infra_img)
        i += 1
        # cv2.imshow("depth", depth_img)
        # cv2.imshow("infra", infra_img)
        cv2.waitKey(1)
