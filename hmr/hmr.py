import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

HMR_INPUT_SIZE = 224
BATCH_SIZE = 1


module_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(module_path, "model/HMR.tflite")
THETA_PATH = os.path.join(module_path, "model/initial_theta.npy")

class HMR:
    
    def __init__(self):        

        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()



        self.initial_theta = np.load(THETA_PATH).reshape(1, 85).repeat(BATCH_SIZE, axis=0).astype(np.float32)
                
        
    def __call__(self, image): 
        image = image / 127.5
        image -= 1.
        image = np.expand_dims(image, 0).astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.set_tensor(self.input_details[1]['index'], self.initial_theta)        
        self.interpreter.invoke()
        hmr_raw = self.interpreter.get_tensor(self.output_details[0]['index'])
    
        return hmr_raw
        
        
        
