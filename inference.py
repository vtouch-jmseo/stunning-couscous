import numpy
import onnxruntime as rt
import tensorflow as tf


class VTouchInference:
    def __init__(self, model_path):
        self.model_path = model_path


class VTouchInferencePB(VTouchInference):
    """docstring for VTouchInferencePB"""
    def __init__(self, model_path):
        super().__init__(model_path)
        self.interpreter = self.load_inference_graph()

    # Load a frozen infrerence graph into memory
    def load_inference_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef() # tf.GraphDef()

            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
        return detection_graph
        
    def run(self, img):
        img = np.array(img, dtype=np.float32)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], img)
        self.interpreter.invoke()
        pred = interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
        pred = tf.reshape(tf.squeeze(pred), [2,1])

        return pred.numpy()

        
class VTouchInferenceOnnx(VTouchInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.sess = rt.InferenceSession(self.model_path)

    # def load_weight(self):
    #     self.sess = rt.InferenceSession(self.model_path)

    def run(self, img):
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[0].name
        pred_onx = self.sess.run(
            [label_name], {input_name: img.astype(numpy.float32)})[0]
        print(pred_onx)


class VTouchInferenceTFLite(VTouchInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.interpreter = None

    def load_weight(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def run(self):
        pass


if __name__ == "__main__":
    pass
