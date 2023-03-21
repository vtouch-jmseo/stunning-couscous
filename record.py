import os
import cv2
import timeit

import numpy as np
import aditofpython as tof

from util import read_yaml, normalize
from raw import ADIFileWriter

RECORD_STATE = False


def on_change(pos):
    pass


def get_camera(conf):
    system = tof.System()

    cameras = []
    status = system.getCameraListAtIp(cameras, conf['ip'])

    print("system.getCameraList()", status)
    print(cameras)
    camera1 = cameras[0]

    status = camera1.setControl("initialization_config", conf['config'])
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

    status = camera1.setFrameType(conf['frame_type'])
    print("camera1.setFrameType()", status)

    # TODO: set noise threshold

    status = camera1.start()
    print("camera1.start()", status)

    return camera1


if __name__ == "__main__":

    conf = read_yaml("./record_conf.yaml")

    if not os.path.isdir(conf['save_prefix']):
        os.mkdir(conf['save_prefix'])

    camera1 = get_camera(conf)

    cv2.namedWindow("Infra Window", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("threshold", "Infra Window", 0, 65000, on_change)
    cv2.setTrackbarPos("threshold", "Infra Window", 3000)
    frame = tof.Frame()

    w = ADIFileWriter(conf['frame_type'], conf['record_time_sec'])

    while True:
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == ord('r'):
            RECORD_STATE = True
            w.generate_file(conf['save_prefix'])

        # Capture frame-by-frame
        status = camera1.requestFrame(frame)
        if not status:
            print("cameras[0].requestFrame() failed with status: ", status)

        depth_map = np.array(frame.getData("depth"), dtype="uint16", copy=True)
        ir_map = np.array(frame.getData("ir"), dtype="uint16", copy=True)
        threshold = cv2.getTrackbarPos("threshold", "Infra Window")

        infra = normalize(ir_map, threshold)
        infra = cv2.cvtColor(infra, cv2.COLOR_GRAY2BGR)
        if RECORD_STATE:
            infra = cv2.putText(infra, "recording", (200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            RECORD_STATE = w.write(ir_map, depth_map)
        cv2.imshow("Infra Window", infra)

    cv2.destroyAllWindows()
    status = camera1.stop()
    print("camera1.close()", status)
