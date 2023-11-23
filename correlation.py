import cv2
import os
import numpy as np
import main


Result_input_path = 'found'
stickers_data = 'stickers_modified'
def correlate(Result_input_path, stickers_data):
    if not os.path.exists(Result_input_path):
        print("No result directory")
        return
    probability = [len(os.listdir(Result_input_path)), len(os.listdir(stickers_data))]
    for sticker_found in os.listdir(Result_input_path):
        image_target = cv2.imread(sticker_found, cv2.IMREAD_UNCHANGED)
        for sticker_base, j in os.listdir(stickers_data):
            sticker = cv2.imread(sticker_base, cv2.IMREAD_UNCHANGED)
            a, b = main.ScalePicture(image_target, sticker).scaleBoth()
            diff = cv2.absdiff(a, b)
            probability[sticker_found, sticker_base] = cv2.countNonZero(cv2.bitwise_not(diff))