# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:05:10 2025

@author: BIDISHA
"""

import cv2
import numpy as np
import os

file_list = []
frame_dict = {}

# Set correct paths
output_frames_path = r"C:\Users\BIDISHA\Desktop\ELL784_Assignment1\Processed_Frames_1DGMM"
video_name = r"C:\Users\BIDISHA\Desktop\ELL784_Assignment1\1d_gmm_expt_1_output_video.mp4"

# Read all frame files
for frame_name in os.listdir(output_frames_path):
    frame_path = os.path.join(output_frames_path, frame_name)
    file_list.append(frame_path)
    frame_num = int(frame_path.split("\\")[-1].split(".")[0])  # Use "\\" for Windows paths
    frame_dict[frame_num] = frame_path

# Read the first frame to get dimensions
sample_frame = cv2.imread(file_list[0])
height, width = sample_frame.shape[:2]

# Sort frame numbers
frame_nums = sorted(frame_dict.keys())

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

# Write frames into video
print("Creating Video ...")
for idx in frame_nums:
    img = cv2.imread(frame_dict[idx])
    video.write(img)

# Save and release
cv2.destroyAllWindows()
video.release()
print("Done! Output video saved at:", video_name)
