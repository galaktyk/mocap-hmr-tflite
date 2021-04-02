import os
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite


inputWidth = 128
inputHeight = 128
iouThreshold = 0.3
AnchorsConfig = [[8,2], [16,6]]
regressor_kps = 4


module_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(module_path, "model/lite_pose_detection.tflite")



class PersonDetector:
    
    def __init__(self):        
        self.input_shape = (inputWidth, inputHeight)     
        
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        
        
        self.anchors = self.gen_anchors()   
        self.anchors_tile = np.tile(self.anchors, regressor_kps).reshape(896, regressor_kps, 2)       
        self.enlarge_box = 1.5
        self.scoreThreshold = 0.3
    

    def gen_anchors(self):
        anchors = []        
        for stride, anchorsNum in AnchorsConfig:        
            gridRows = (inputHeight + stride - 1) // stride
            gridCols = (inputWidth + stride - 1) // stride        
            
            for gridY in range(gridRows) :
                anchorY = stride * (gridY + 0.5)
                for gridX in range(gridCols):
                    anchorX = stride * (gridX + 0.5)                
                    for n in range(anchorsNum):
                        anchors.append([anchorX, anchorY])
        return np.array(anchors)
    
    
    def preprocess_input(self, image):    
        image = cv2.resize(image, self.input_shape).astype('float')             
        image /= 127.5
        image -= 1.        
        return np.expand_dims(image, 0)
    
    
    def decodeBounds(self, regressors_output, max_idx):        
       
        landmarks_multi = regressors_output.reshape(896, 4, 2) + self.anchors_tile
        landmarks = landmarks_multi[max_idx]
        
        
        # Key point 0 - mid shoulder center
        # Key point 1 - point that encodes size & rotation (upper body)
        # Key point 2 - mid hip center
        # Key point 3 - point that encodes size & rotation (full body)        
        center_xy, head_xy = landmarks[2], landmarks[3]
        half_box_size = np.linalg.norm(center_xy - head_xy) * self.enlarge_box 
        x1, y1 = center_xy - half_box_size
        x2, y2 = center_xy + half_box_size
        

        return np.array([y1, x1, y2, x2])
    
        
    def __call__(self, image):
        in_h, in_w, _ = image.shape
        assert in_h == in_w
        ratio = in_h/inputHeight
        
        
        net_input = self.preprocess_input(image).astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], net_input)
        self.interpreter.invoke()
        classifiers_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        regressors_output = self.interpreter.get_tensor(self.output_details[1]['index'])
           

        # Parse only 1 person.
        scores = classifiers_output.reshape(-1)
        max_idx = scores.argmax()
        
        # Get box around person.
        box = self.decodeBounds(regressors_output.squeeze(), max_idx)       
   
        return (box * ratio).astype('int')
        

    def non_max(self, decodedBounds, landmarks, scores):
        selected_ids = tf.image.non_max_suppression(decodedBounds, scores,
                                                    max_output_size=10,
                                                    iou_threshold=0.7,
                                                    score_threshold=self.scoreThreshold)
        
        selected_ids = selected_ids.numpy()
        print(selected_ids)
        
        return decodedBounds[selected_ids], landmarks[selected_ids]
    
    


