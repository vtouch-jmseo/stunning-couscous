import os
import cv2
import time
import numpy as np
import aditofpython as tof

from util import read_yaml, normalize
from inference import VTouchInferenceOnnx, VTouchInferenceTFLite, VTouchInferencePB
from raw import ADIFileReader, ADICameraReader

IR_NORM_VALUE = 200
DEPTH_NORM_VALUE = 4000


def get_concat(infra, depth):
    zero_img = np.zeros(infra.shape, dtype=np.uint8)

    return np.dstack((depth, infra, zero_img))


if __name__ == "__main__":

    conf = read_yaml("./record_conf.yaml")

    if conf['mode']:
        camera = ADICameraReader(conf)
    else:
        camera = ADIFileReader(conf['file_path'])

    # cv2.namedWindow("Infra Window", cv2.WINDOW_NORMAL)

    model = VTouchInferencePB("./frozen_inference_graph.pb")
    # model = VTouchInferenceTFLite("./tflite_graph.tflite")
    # model = VTouchInferenceOnnx("./re_model.onnx")
    
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

        if camera.is_eof():
            break
    
        start_time_fps = cv2.getTickCount()
        start_time = time.time()

        ir_map, depth_map = camera.get_frame()

        infra = normalize(ir_map, IR_NORM_VALUE)
        depth = normalize(depth_map, DEPTH_NORM_VALUE)
        concat = get_concat(infra, depth)
        concat = cv2.cvtColor(concat, cv2.COLOR_BGR2RGB)
        pred = model.run(concat)
        frame = concat.copy()
        

        cv2.imshow("Infra Window", concat)

        end_time = time.time()
        end_time_fps = cv2.getTickCount()

        elapsed_time = (end_time_fps - start_time_fps)
        fps = cv2.getTickFrequency() /elapsed_time

        show_image = concat.copy()
        
    cv2.destroyAllWindows()
    camera.close()
