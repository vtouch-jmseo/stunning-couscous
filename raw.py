import os
import cv2
import struct

import numpy as np
from datetime import datetime


class ADIFileWriter:
    def __init__(self, frame_type, record_time):
        self.frame_type = frame_type
        self.record_time = record_time
        self.fps = 7 if frame_type == 'mp' else 30
        self.shape = (1024, 1024) if frame_type == 'mp' else (512, 512)
        self.num_frames = 0
        self.f = None

    def generate_file(self, save_prefix):
        now = datetime.now()
        date = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"{date}.adi"

        file_path = os.path.join(save_prefix, file_name)
        self.f = open(file_path, "wb")

        header = struct.pack('III', self.shape[0], self.shape[1], self.fps * self.record_time)
        self.f.write(header)

    def write(self, ir_map, depth_map):
        if self.num_frames < self.fps * self.record_time:
            flatten_infra = ir_map.flatten()
            flatten_depth = depth_map.flatten()
            fmt = "H" * len(flatten_depth)

            pack_infra = struct.pack(fmt, *flatten_infra)
            pack_depth = struct.pack(fmt, *flatten_depth)

            self.f.write(pack_infra)
            self.f.write(pack_depth)
            self.num_frames += 1
            state = True
        else:
            self.f.close()
            self.num_frames = 0
            state = False

        return state


class ADIFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = open(self.file_path, 'rb')
        self.shape = [struct.unpack('I', self.f.read(struct.calcsize('I')))[0] for _ in range(3)]

    def read(self):
        infra = [struct.unpack('H', self.f.read(struct.calcsize('H')))[0] for _ in range(self.shape[0] * self.shape[1])]
        depth = [struct.unpack('H', self.f.read(struct.calcsize('H')))[0] for _ in range(self.shape[0] * self.shape[1])]
        infra = np.array(infra).reshape(self.shape[1], self.shape[0])
        depth = np.array(depth).reshape(self.shape[1], self.shape[0])

        return infra, depth

    def close(self):
        self.f.close()

    def get_shape(self):
        return self.shape

