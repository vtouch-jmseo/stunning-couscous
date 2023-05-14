import cv2 

import numpy as np
# import onnxruntime as rt
import tensorflow as tf


class VTouchInference:
    def __init__(self, model_path):
        self.model_path = model_path


class VTouchInferencePB(VTouchInference):
    """docstring for VTouchInferencePB"""
    def __init__(self, model_path):
        super().__init__(model_path)
        self.detection_graph = self.load_inference_graph()
        self.sess = tf.Session(graph=self.detection_graph)
        self.detect_tensor_name = ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 
                                    'num_detections:0', 'detection_keypoints:0']

    # Load a frozen infrerence graph into memory
    def load_inference_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()

            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
        return detection_graph
        
    def run(self, img):
        img = np.expand_dims(img, axis=0)

        input_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        output_tensors = [self.detection_graph.get_tensor_by_name(name) for name in self.detect_tensor_name]

        (boxes, scores, classes, num, keypoints) = self.sess.run(output_tensors, feed_dict={input_tensor: img})
        
        return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes). keypoints[0]

        
class VTouchInferenceOnnx(VTouchInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.sess = rt.InferenceSession(self.model_path)

    def run(self, img):
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[0].name
        pred_onx = self.sess.run(
            [label_name], {input_name: img.astype(np.float32)})[0]
        print(pred_onx)


class VTouchInferenceTFLite(VTouchInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.load_weight()

    def load_weight(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def run(self, img):
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()

        frame = cv2.resize(img, (200, 200))
        batch = np.asarray([frame for i in range(1)])
        batch = batch.astype(np.float32)
        batch = batch / 255.0
        scale, zero_point = input_details['quantization']
        batch = batch / scale + zero_point
        batch = batch.astype(np.uint8)
        print(batch.shape)

        self.interpreter.set_tensor(input_details['index'], batch)
        self.interpreter.invoke()

        for output in output_details:
            tmp = self.interpreter.get_tensor(output['index'])
            print(tmp)


if __name__ == "__main__":
    pass