import cv2
import os
import numpy as np

Result_input_path = 'found'
stickers_data = 'stickers_modified'


def correlate(Result_input_path, stickers_data):
    if not os.path.exists(Result_input_path):
        print("No result directory")
        return

    sticker_list = os.listdir(stickers_data)
    num_stickers = len(sticker_list)

    probability = np.zeros((len(os.listdir(Result_input_path)), num_stickers))
    sticker_names = np.empty((len(os.listdir(Result_input_path)), num_stickers), dtype=object)

    i = 0
    for sticker_found in os.listdir(Result_input_path):
        image_target = cv2.imread(sticker_found, cv2.IMREAD_UNCHANGED)
        j = 0
        for sticker_base in sticker_list:
            sticker_path = os.path.join(stickers_data, sticker_base)
            sticker_name = os.path.basename(sticker_path)
            sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
            a, b = main.ScalePicture(image_target, sticker).scaleBoth()
            diff = cv2.absdiff(a, b)
            diff_height, diff_width = diff.shape[:2]
            probability[i][j] = cv2.countNonZero(cv2.bitwise_not(diff)) / (diff_width * diff_height)
            sticker_names[i][j] = sticker_name
            j += 1
        i += 1

    return probability, sticker_names


def saver_drawer(probability, sticker_names):
    i = 0
    for sticker_found in os.listdir(Result_input_path):
        image_target = cv2.imread(sticker_found, cv2.IMREAD_UNCHANGED)
        max_index = np.argmax(probability[i])
        max_prob = probability[i][max_index]
        sticker_name = sticker_names[i][max_index]
        cv2.imwrite(f"finally/{i}_{sticker_name}_{max_prob}.jpg", image_target)
        cv2.imshow(f"{i}_{sticker_name}_{max_prob}", image_target)
        i += 1
