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






