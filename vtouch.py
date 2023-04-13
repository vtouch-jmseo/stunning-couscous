import os
import cv2
# print(os.environ.get('LD_LIBRARY_PATH', None))
import numpy as np
import aditofpython as tof

from util import read_yaml, normalize
from inference import VTouchInferenceOnnx, VTouchInferencePB

IR_NORM_VALUE = 200
DEPTH_NORM_VALUE = 2000


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


def get_concat(infra, depth):
    zero_img = np.zeros(infra.shape, dtype=np.uint8)

    return np.dstack((depth, infra, zero_img))


if __name__ == "__main__":

    conf = read_yaml("./record_conf.yaml")
    camera1 = get_camera(conf)

    cv2.namedWindow("Infra Window", cv2.WINDOW_NORMAL)
    # cv2.setTrackbarPos("threshold", "Infra Window", 3000)

    frame = tof.Frame()

    # model = VTouchInferenceOnnx("/home/jeongmin/Downloads/vtouch_mobilenetv2_face_04to0511to131820to21_epoch100.onnx")
    model = VTouchInferencePB("/home/jeongmin/Desktop/adi_model/frozen_inference_graph.pb")

    while True:
        key = cv2.waitKey(1)

        if key == 27:
            break
    
        # Capture frame-by-frame
        status = camera1.requestFrame(frame)
        if not status:
            print("cameras[0].requestFrame() failed with status: ", status)

        depth_map = np.array(frame.getData("depth"), dtype="uint16", copy=True)
        ir_map = np.array(frame.getData("ir"), dtype="uint16", copy=True)
        # threshold = cv2.getTrackbarPos("threshold", "Infra Window")

        infra = normalize(ir_map, IR_NORM_VALUE)
        depth = normalize(depth_map, DEPTH_NORM_VALUE)
        concat = get_concat(infra, depth)

        # infra = cv2.cvtColor(infra, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Infra Window", concat)
    
    cv2.destroyAllWindows()
    status = camera1.stop()
    print("camera1.close()", status)
