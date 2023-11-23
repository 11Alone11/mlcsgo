import cv2
import os
import math
import numpy as np
from shapely.geometry import box


class ScalePicture:
    def __init__(self, img_1, img_2=None):
        self.img_1 = img_1
        self.img_2 = img_2
        self.scaled_img_1 = None
        self.scaled_img_2 = None

    def scaleFirst(self):
        scale_percent = 50  # Процент от исходного размера
        width = int(self.img_1.shape[1] * scale_percent / 100)
        height = int(self.img_1.shape[0] * scale_percent / 100)
        dim = (width, height)
        scaled_img_1 = cv2.resize(self.img_1, dim, interpolation=cv2.INTER_AREA)
        self.scaled_img_1 = scaled_img_1
        return self.scaled_img_1

    def scaleSecond(self):
        # if(self.img_2 == None):
        #    print("No img_2 value")
        #    return None
        scale_percent = 50  # Процент от исходного размера
        width = int(self.img_2.shape[1] * scale_percent / 100)
        height = int(self.img_2.shape[0] * scale_percent / 100)
        dim = (width, height)
        scaled_img_2 = cv2.resize(self.img_2, dim, interpolation=cv2.INTER_AREA)
        self.scaled_img_2 = scaled_img_2
        return self.scaled_img_2

    def scaleBoth(self):
        # if (self.img_2 == None):
        #    print("No img_2 value")
        #    return None, None
        width = int((self.img_1.shape[1] + self.img_2.shape[1]) / 2)
        height = int((self.img_1.shape[0] + self.img_2.shape[0]) / 2)
        dim = (width, height)
        scaled_ea_1 = cv2.resize(self.img_1, dim, interpolation=cv2.INTER_AREA)
        scaled_ea_2 = cv2.resize(self.img_2, dim, interpolation=cv2.INTER_AREA)
        return scaled_ea_1, scaled_ea_2


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
    # print(f"intersect_y: {intersect_y} , coord_x_r1 : {r1[0]}, coord_x_r2 : {r2[0]}")
    # print(intersect_x)
    return rect1.intersects(rect2) and not (intersect_x < stock or intersect_y < stock)


def area(r):
    """Возвращает площадь прямоугольника r."""
    _, _, w, h = r
    return w * h


##################################################################################################

# Загрузка изображений

original_img = cv2.imread('Images/h4.jpg')
edited_img1 = cv2.imread('Images/h3.jpg')

original_img, edited_img1 = ScalePicture(original_img, edited_img1).scaleBoth()

# Изменение размера edited_img для соответствия размеру original_img
if np.any(original_img[0] != edited_img1[0]) and np.any(original_img[1] != edited_img1[1]):
    edited_img1 = cv2.resize(edited_img1, (original_img.shape[1], original_img.shape[0]))

# Нахождение различий
diff = cv2.absdiff(original_img, edited_img1)

alfa = ScalePicture(diff).scaleFirst()
cv2.imshow("Difference", alfa)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

# Проверка на вложенность и пересечение
for r in rects:
    if not any(is_inside(r, other) for other in rects if r != other) and \
            not any(intersects(r, other) and area(other) > area(r) for other in rects if r != other)\
            and ((r[0] > 0.7 * r[1] or r[0] > 1.3 * r[1]) and (r[0] > 0.7 * math.sqrt(area(r)) or r[0] > 1.3 * math.sqrt(area(r)))):
        final_rects.append(r)

#and ((r[0] > 1000 and r[1] > 1000) or ((600 < r[0] < 1200) and (100 < r[1] < 300))) <<< buff staff

save_path = 'found'
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
        cv2.rectangle(original_img, start_position, finish_position, rectangle_color,
                      rectangle_line_thickness)  # Отрисовка прямоугольника

#Сохранить результат
cv2.imwrite(f"{save_path}/detected_stickers.jpg", original_img)

# Показать измененное изображение
resized_image = ScalePicture(original_img).scaleFirst()
cv2.imshow("Output", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

