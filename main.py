# import cv2
# import numpy as np
#
# # Загрузка изображений
# sticker_img = cv2.imread('Images/ibp_money.png', cv2.IMREAD_GRAYSCALE)
# weapon_screenshot = cv2.imread('Images/ibp_ak.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Инициализация ORB
# orb = cv2.ORB_create()
#
# # Находим ключевые точки и дескрипторы
# keypoints1, descriptors1 = orb.detectAndCompute(sticker_img, None)
# keypoints2, descriptors2 = orb.detectAndCompute(weapon_screenshot, None)
#
# # Сопоставление дескрипторов
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
#
# # Сортировка совпадений по расстоянию
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Отображение совпадений
# matched_img = cv2.drawMatches(sticker_img, keypoints1, weapon_screenshot, keypoints2, matches[:100], None, flags=2)
#
# scale_percent = 50  # Процент от исходного размера
# width = int(matched_img.shape[1] * scale_percent / 100)
# height = int(matched_img.shape[0] * scale_percent / 100)
# dim = (width, height)
#
# # Изменение размера изображения
# resized_img = cv2.resize(matched_img, dim, interpolation = cv2.INTER_AREA)
#
# # Показать измененное изображение
# cv2.imshow("Output", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # cv2.imshow('Matches', matched_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# import cv2
# import matplotlib.pyplot as plt
#
# # Загрузка изображений
# sticker_img = cv2.imread('Images/ibp_money.png', cv2.IMREAD_GRAYSCALE)
# weapon_screenshot = cv2.imread('Images/ibp_ak.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Создание детектора ORB
# orb = cv2.ORB_create(nfeatures=1000)  # Увеличение числа ключевых точек
#
#
# # Детектирование ключевых точек и вычисление дескрипторов
# keypoints1, descriptors1 = orb.detectAndCompute(sticker_img, None)
# keypoints2, descriptors2 = orb.detectAndCompute(weapon_screenshot, None)
#
# # Сопоставление признаков
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
# matches = sorted(matches, key=lambda x: x.distance)
#
# # Визуализация сопоставления
# matched_img = cv2.drawMatches(sticker_img, keypoints1, weapon_screenshot, keypoints2, matches[:100], None, flags=2)
# plt.imshow(matched_img)
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Загрузка изображений
# sticker_img = cv2.imread('Images/ibp_money.png', cv2.IMREAD_GRAYSCALE)
# weapon_screenshot = cv2.imread('Images/ibp_2.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Инициализация SIFT
# sift = cv2.SIFT_create()
#
# # Нахождение ключевых точек и дескрипторов с SIFT
# keypoints1, descriptors1 = sift.detectAndCompute(sticker_img, None)
# keypoints2, descriptors2 = sift.detectAndCompute(weapon_screenshot, None)
#
# # Использование FLANN для сопоставления дескрипторов
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(descriptors1, descriptors2, k=2)
#
# # Lowe's ratio test для фильтрации хороших сопоставлений
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.8 * n.distance:
#         good_matches.append(m)
#
# # Нахождение гомографии
# if len(good_matches) > 4:
#     src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
#     dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#
#     # Визуализация
#     draw_params = dict(matchColor = (0,255,0),
#                        singlePointColor = None,
#                        matchesMask = matchesMask,
#                        flags = 2)
#     matched_img = cv2.drawMatches(sticker_img, keypoints1, weapon_screenshot, keypoints2, good_matches, None, **draw_params)
#     plt.imshow(matched_img, 'gray')
#     plt.show()
# else:
#     print("Недостаточно хороших сопоставлений - %d/%d" % (len(good_matches), 4))



# import cv2
#
# # Загрузка изображений
# sticker_img = cv2.imread('Images/ibp_money.png', cv2.IMREAD_GRAYSCALE)
# weapon_screenshot = cv2.imread('Images/ibp_ak.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Создание BRISK детектора
# brisk = cv2.BRISK_create()
#
# # Нахождение ключевых точек и дескрипторов
# keypoints1, descriptors1 = brisk.detectAndCompute(sticker_img, None)
# keypoints2, descriptors2 = brisk.detectAndCompute(weapon_screenshot, None)
#
# # Сопоставление с использованием KNN
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Отображение результатов
# matched_img = cv2.drawMatches(sticker_img, keypoints1, weapon_screenshot, keypoints2, matches[:10], None, flags=2)
#
#
# scale_percent = 50  # Процент от исходного размера
# width = int(matched_img.shape[1] * scale_percent / 100)
# height = int(matched_img.shape[0] * scale_percent / 100)
# dim = (width, height)
#
# # Изменение размера изображения
# resized_img = cv2.resize(matched_img, dim, interpolation = cv2.INTER_AREA)
#
# # Показать измененное изображение
# cv2.imshow("Output", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
#
# # Загрузка изображений
# original_img = cv2.imread('Images/ibp_ak.jpg') # Замените на ваш путь к оригинальному изображению
# edited_img = cv2.imread('Images/unc.jpg') # Замените на ваш путь к измененному изображению
#
# # Вычисление разницы
# diff = cv2.absdiff(original_img, edited_img)
# gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Выделение областей на оригинальном изображении
# for contour in contours:
#     if cv2.contourArea(contour) > 100:  # Минимальный размер области
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
# # Показать результат
# cv2.imshow('Detected Stickers', original_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import os
import numpy as np
from shapely.geometry import box

def is_inside(r1, r2):
    """Проверяет, содержится ли r1 (x, y, w, h) полностью внутри r2."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 >= x2) and (y1 >= y2) and (x1 + w1 <= x2 + w2) and (y1 + h1 <= y2 + h2)

def intersects(r1, r2):
    """Проверяет, пересекаются ли прямоугольники r1 и r2."""
    rect1 = box(r1[0], r1[1], r1[0] + r1[2], r1[1] + r1[3])
    rect2 = box(r2[0], r2[1], r2[0] + r2[2], r2[1] + r2[3])
    return rect1.intersects(rect2)

def area(r):
    """Возвращает площадь прямоугольника r."""
    _, _, w, h = r
    return w * h

# Загрузка изображений
original_img = cv2.imread('Images/m1.jpg') # Замените на ваш путь к оригинальному изображению
edited_img1 = cv2.imread('Images/m0.jpg') # Замените на ваш путь к измененному изображению
edited_img2 = cv2.imread('Images/m2.jpg') # Замените на ваш путь к измененному изображению
edited_img3 = cv2.imread('Images/m3.jpg') # Замените на ваш путь к измененному изображению


# Изменение размера edited_img для соответствия размеру original_img
# edited_img = cv2.resize(edited_img, (original_img.shape[1], original_img.shape[0]))


# Нахождение различий
diff = cv2.absdiff(original_img, edited_img1)
diff2 = cv2.absdiff(edited_img3, edited_img1)
diff3 = cv2.absdiff(original_img, edited_img2)
diff4 = cv2.absdiff(original_img, edited_img3)

diffa = cv2.absdiff(diff, diff2)
diffb = cv2.absdiff(diff3, diff4)

diff = cv2.absdiff(diffa, diffb)


gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

# Поиск контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Пороговая площадь для фильтрации маленьких контуров
area_threshold = 1000

# Получение прямоугольников для каждого контура
rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > area_threshold]

# Удаление вложенных и пересекающихся прямоугольников
min_width = 0  # минимальная ширина
min_height = 0  # минимальная высота
final_rects = []

for r in rects:
    # Проверка на вложенность и пересечение
    if not any(is_inside(r, other) for other in rects if r != other) and \
       not any(intersects(r, other) and area(other) > area(r) for other in rects if r != other):
        final_rects.append(r)

save_path = 'dataset'
os.makedirs(save_path, exist_ok=True)

# Вырезание и сохранение областей с наклейками
for i, (x, y, w, h) in enumerate(final_rects):
    if w >= min_width and h >= min_height:
        roi = original_img[y:y + h, x:x + w]
        cv2.imwrite(f"{save_path}/sticker_{i}.jpg", roi)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Отрисовка прямоугольника

# Показать и сохранить результат
cv2.imshow('Detected Stickers', original_img)
cv2.imwrite(f"{save_path}/detected_stickers.jpg", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

scale_percent = 30  # Процент от исходного размера
width = int(original_img.shape[1] * scale_percent / 100)
height = int(original_img.shape[0] * scale_percent / 100)
dim = (width, height)

# Изменение размера изображения
resized_img = cv2.resize(original_img, dim, interpolation = cv2.INTER_AREA)

# Показать измененное изображение
cv2.imshow("Output", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()






