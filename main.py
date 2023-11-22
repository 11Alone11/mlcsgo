import cv2
import os
import math
import numpy as np
from shapely.geometry import box

class ScalePicture:
    def __init__(self, img):
        self.img = img
        self.scaled_img = None
    def scale(self):
        scale_percent = 50  # Процент от исходного размера
        width = int(self.img.shape[1] * scale_percent / 100)
        height = int(self.img.shape[0] * scale_percent / 100)
        dim = (width, height)
        scaled_img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)
        self.scaled_img = scaled_img
        return self.scaled_img

def is_inside(r1, r2):
    """Проверяет, содержится ли r1 (x, y, w, h) полностью внутри r2."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 >= x2) and (y1 >= y2) and (x1 + w1 <= x2 + w2) and (y1 + h1 <= y2 + h2)

def intersects(r1, r2):
    """Проверяет, пересекаются ли прямоугольники r1 и r2."""
    intersect_y = 0.
    intersect_x = 0.
    stock = 1
    rect1 = box(r1[0], r1[1], r1[0] + r1[2], r1[1] + r1[3])
    rect2 = box(r2[0], r2[1], r2[0] + r2[2], r2[1] + r2[3])

    up_two_one_down = r1[1] < r2[1] + r2[3] and r1[1] + r1[3] > r2[1]
    up_one_two_down = r2[1] < r1[1] + r1[3] and r2[1] + r2[3] > r1[1]
    if up_two_one_down:
        intersect_y = min(math.fabs(r1[1] - r2[1] - r2[3]), math.fabs(r1[1] + r1[3] - r2[1]))
    elif up_one_two_down:
        intersect_y = min(math.fabs(r2[1] - r1[1] - r1[3]), math.fabs(r2[1] + r2[3] - r1[1]))

    left_one_two_right = r1[0] < r2[0] + r2[2] and r1[0] + r1[2] > r2[0]
    left_two_one_right = r2[0] < r1[0] + r1[2] and r2[0] + r2[2] > r1[0]
    if left_one_two_right:
        intersect_x = min(math.fabs(r1[0] - r2[0] - r2[2]), math.fabs(r1[0] + r1[2] - r2[0]))
    elif left_two_one_right:
        intersect_x = min(math.fabs(r2[0] - r1[0] - r1[2]), math.fabs(r2[0] + r2[2] - r1[0]))
    #print(f"intersect_y: {intersect_y} , coord_x_r1 : {r1[0]}, coord_x_r2 : {r2[0]}")
    #print(intersect_x)
    return rect1.intersects(rect2) and not(intersect_x < stock or intersect_y < stock)

def area(r):
    """Возвращает площадь прямоугольника r."""
    _, _, w, h = r
    return w * h
##################################################################################################

# Загрузка изображений
original_img = cv2.imread('Images/h8.jpg')
edited_img1 = cv2.imread('Images/h7.jpg')
height1, width1, _ = original_img.shape
height1 = math.floor(height1/2)
width1 = math.floor(width1/2)


original_img = cv2.resize(original_img, (2000, 1600), interpolation=cv2.INTER_AREA)
edited_img1 = cv2.resize(edited_img1, (2000, 1600), interpolation=cv2.INTER_AREA)

#cv2.imshow("orig", original_img)
#cv2.imshow("edit", edited_img1)


# Изменение размера edited_img для соответствия размеру original_img
if np.any(original_img[0] != edited_img1[0]) and np.any(original_img[1] != edited_img1[1]):
    edited_img1 = cv2.resize(edited_img1, (original_img.shape[1], original_img.shape[0]))


# Нахождение различий
diff = cv2.absdiff(original_img, edited_img1)

# Смена цветовой палитры таргета
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

# Поиск контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
       not any(intersects(r, other) and area(other) > area(r) for other in rects if r != other)\
            and ((r[0] > 1000 and r[1] > 1000) or ((600 < r[0] < 1200) and (100 < r[1] < 300))):
        final_rects.append(r)

save_path = 'dataset'
os.makedirs(save_path, exist_ok=True)

# Вырезание и сохранение областей с наклейками
rectangle_color = (255, 235, 59)
rectangle_line_thickness = 2
for i, (x, y, w, h) in enumerate(final_rects):
    if w >= min_width and h >= min_height:
        roi = original_img[y:y + h, x:x + w]
        cv2.imwrite(f"{save_path}/sticker_{i}.jpg", roi)
        start_position = (x, y)
        finish_position = (x + w, y + h)
        cv2.rectangle(original_img, start_position, finish_position, rectangle_color, rectangle_line_thickness)  # Отрисовка прямоугольника

# Показать и сохранить результат
#cv2.imshow('Detected Stickers', original_img)
cv2.imwrite(f"{save_path}/detected_stickers.jpg", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

scaler = ScalePicture(original_img)
resized_image = scaler.scale()

# Показать измененное изображение
cv2.imshow("Output", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()






