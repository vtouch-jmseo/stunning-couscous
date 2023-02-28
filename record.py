import os
import sys
import cv2
import timeit
import struct

import numpy as np
import aditofpython as tof
from datetime import datetime

IP = "10.42.0.1"
CONFIG = "./config/config_walden_3500_nxp.json"

# from video_writer import VideoWriter
SAVE_PREFIX = "./record"
RECORD_TIME_SEC = 3
RECORD_STATE = False
FRAME_TYPE = 'lrmp'
shape = (1024, 1024) if FRAME_TYPE == 'lrmp' else (512, 512)
fps = 5 if FRAME_TYPE == 'lrmp' else 10


def normalize(mat, threshold):
    # mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    mat[mat>threshold] = threshold
    mat = mat / threshold * 255
    mat = np.asarray(mat, dtype=np.uint8)

    return mat


def generate_filename():
    now = datetime.now()
    date = now.strftime("%Y%m%d_%H%M%S")

    return f"{date}.adi"


def on_change(pos):
    pass


if __name__ == "__main__":

    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)

    system = tof.System()

    cameras = []
    status = system.getCameraListAtIp(cameras, IP)

    print("system.getCameraList()", status)
    print(cameras)
    camera1 = cameras[0]

    status = camera1.setControl("initialization_config", CONFIG)
    print("camera1.setControl()", status)

    status = camera1.initialize()
    print("camera1.initialize()", status)

    camDetails = tof.CameraDetails()
    status = camera1.getDetails(camDetails)
    print("system.getDetails()", status)
    print("camera1 details:", "id:", camDetails.cameraId, "connection:", camDetails.connection)

    types = []
    status = camera1.getAvailableFrameTypes(types)
    print("system.getAvailableFrameTypes()", status)
    print(types)

    status = camera1.setFrameType('lrmp')
    print("camera1.setFrameType()", status)

    # TODO: set noise threshold

    status = camera1.start()
    print("camera1.start()", status)

    cv2.namedWindow("Infra Window")
    cv2.createTrackbar("threshold", "Infra Window", 0, 65000, on_change)
    cv2.setTrackbarPos("threshold", "Infra Window", 65000)
    frame = tof.Frame()
    record_start = timeit.default_timer()

    num_cur_record_frame = 0
    while True:
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == ord('r'):
            RECORD_STATE = True

            file_name = generate_filename()
            file_path = os.path.join(SAVE_PREFIX, file_name)
            f = open(file_path, "wb")
            header = struct.pack('III', shape[0], shape[1], fps * RECORD_TIME_SEC)
            f.write(header)

        # Capture frame-by-frame
        status = camera1.requestFrame(frame)
        if not status:
            print("cameras[0].requestFrame() failed with status: ", status)

        depth_map = np.array(frame.getData("depth"), dtype="uint16", copy=True)
        ir_map = np.array(frame.getData("ir"), dtype="uint16", copy=True)
        threshold = cv2.getTrackbarPos("threshold", "Infra Window")

        infra = normalize(ir_map, threshold)
        infra = np.roll(infra, -5)  # TODO: remove
        infra = cv2.cvtColor(infra, cv2.COLOR_GRAY2BGR)
        if RECORD_STATE:
            infra = cv2.putText(infra, "recording", (200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Infra Window", infra)

        if RECORD_STATE:
            if num_cur_record_frame < fps * RECORD_TIME_SEC:
                flatten_infra = ir_map.flatten()
                flatten_depth = depth_map.flatten()
                fmt = "H" * len(flatten_depth)

                pack_infra = struct.pack(fmt, *flatten_infra)
                pack_depth = struct.pack(fmt, *flatten_depth)

                f.write(pack_infra)
                f.write(pack_depth)
                num_cur_record_frame += 1
            else:
                f.close()
                num_cur_record_frame = 0
                RECORD_STATE = False

    cv2.destroyAllWindows()
    status = camera1.stop()
    print("camera1.close()", status)

    # file_name = generate_filename()
    # file_path = os.path.join(SAVE_PREFIX, file_name)
