import csv
import os
import sys
import bpy

input_csv = sys.argv[7]
target_file = sys.argv[8]

objects = bpy.context.scene.objects

empties = []

for object in objects:
    if object.type == 'EMPTY':
        empties.append(object)
        
with open(input_csv, 'r', newline='') as csvfile:
    ofile = csv.reader(csvfile, delimiter=',')
    next(ofile) # <-- skip the x,y,z header

    for frame_num, pts in enumerate(ofile): 
        # these things are still strings (that's how they get stored in the file)
        # here we recast them to integer and floats
   
        fpts = [float(p) for p in pts]
        coordinates = [fpts[0:3], fpts[3:6], fpts[6:9], fpts[9:12],
                       fpts[12:15], fpts[15:18], fpts[18:21], fpts[21:24],
                       fpts[24:27], fpts[27:30], fpts[30:33], fpts[33:36],
                       fpts[36:39], fpts[39:42],fpts[42:45], fpts[45:48],
                       fpts[48:51], fpts[51:54], fpts[54:57], fpts[57:60]]
                       

        bpy.context.scene.frame_set(frame_num)
        bpy.data.scenes['Scene'].frame_end = frame_num + 1
        
        for ob, position in zip(empties, coordinates):
            ob.location = position
            ob.keyframe_insert(data_path="location", index=-1)



bpy.data.objects['rig'].select_set(True)
bpy.ops.export_anim.bvh(filepath=target_file)
