import sys
import cv2
from tkinter import filedialog, simpledialog
from tkinter import *

from np_smpl.batch_smpl import SMPL
from person_detector import PersonDetector
from hmr import HMR
from bvh_exporter import BVHExporter
from crop_tool import crop_and_resize

root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename()


person_det = PersonDetector()
hmr_detector = HMR()    
smpl_resolver = SMPL()
bvh = BVHExporter()
HMR_INPUT_SIZE = 224


cap = cv2.VideoCapture(video_path)


def pad_square(image):        
    h, w, _ = frame.shape
    pad = abs((h - w)) // 2
    if h > w:        
        top, bottom = 0, 0
        left, right = pad, pad
    else:   
        left, right = 0, 0
        top, bottom = pad, pad               
    
    return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT)

  


while True:
    ret, frame = cap.read()
    if not ret:
        break                
    
    frame = pad_square(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect one person in image.
    person_box = person_det(frame_rgb) 
    cv2.rectangle(frame,
                  tuple(person_box[[1, 0]]), tuple(person_box[[3, 2]]), 
                  (0,0,255), 3)           
     
    # Crop an image.    
    person_rgb = crop_and_resize(frame_rgb, person_box, HMR_INPUT_SIZE)
    
  
    # Run HMR.
    hmr_raw = hmr_detector(person_rgb)
  
    
    # Decode HMR output to 3D info.
    result_dict = smpl_resolver.get_details(hmr_raw)
    joints3d = result_dict['j3d']
    
    
    # Add this frame joints.
    bvh.add(joints3d)
    
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break



    
cap.release()
cv2.destroyAllWindows()


bvh.dump('result/result.bvh')
