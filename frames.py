# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:19:20 2025

@author: BIDISHA
"""

import cv2
import os

# Path to video file
vid = cv2.VideoCapture("C:\\Users\\BIDISHA\\Downloads\\umcp.mp4")
print("Finished reading video!")

# Path to save frames
save_path = "C:/Users/BIDISHA/Desktop/ELL784_Assignment1/frames/"

# Ensure the frames folder exists
os.makedirs(save_path, exist_ok=True)

count = 0
flag = True
while flag:
    flag, frame = vid.read()

    # Break if no more frames
    if not flag or frame is None:
        print("All frames extracted successfully!")
        break

    print("frame: ", frame)

    # Save frames with proper path construction
    cv2.imwrite(save_path + str(count) + ".bmp", frame)

    count += 1

print("Extraction of frames completed!")