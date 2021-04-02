import subprocess
import pandas as pd
import os
import ctypes

JOINTS_NAME = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
            'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
            'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
            'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
            'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
            'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
            'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
            'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
            'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
            'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
            'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
            'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
            'Neck_x', 'Neck_y', 'Neck_z', 
            'Head_x', 'Head_y', 'Head_z', 
            'Nose_x', 'Nose_y', 'Nose_z', 
            'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
            'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
            'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
            'Ear.R_x', 'Ear.R_y', 'Ear.R_z']



class BVHExporter:

    def __init__(self):
        self.df_list = []
                
        
    def add(self, joints3d):        
        joints_info = pd.DataFrame(joints3d.reshape(1, 57), columns=JOINTS_NAME)
        
     
        joints_info.iloc[:, 1::3] = joints_info.iloc[:, 1::3]*-1
        joints_info.iloc[:, 2::3] = joints_info.iloc[:, 2::3]*-1        
        hipCenter = joints_info.loc[:][['Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                                        'Hip.L_x', 'Hip.L_y', 'Hip.L_z']]

        joints_info['hip.Center_x'] = hipCenter.iloc[0][::3].sum()/2
        joints_info['hip.Center_y'] = hipCenter.iloc[0][1::3].sum()/2
        joints_info['hip.Center_z'] = hipCenter.iloc[0][2::3].sum()/2
        
        self.df_list.append(joints_info.copy())
        
        
        
    def dump(self, bvh_path):
        concatenated_df = pd.concat(self.df_list, ignore_index=True)

        if not os.path.isdir('result'):
            os.mkdir('result')

        # Save .csv file.
        csv_path = bvh_path.replace('.bvh', '.csv')
        concatenated_df.to_csv(csv_path, index=False)

        # Python subprocess script that run Blender that run Python script with Blender's python API.
        try:
            subprocess.call(['blender', '--background', 'blender_scripts/csv_to_bvh.blend', '-noaudio', '-P', 'blender_scripts/csv_to_bvh.py', '--', csv_path, bvh_path], shell=True)
        except:         
            ctypes.windll.user32.MessageBoxW(0,  u"Blender not found, install Blender and add to your PATH first.",u"Error", 0)