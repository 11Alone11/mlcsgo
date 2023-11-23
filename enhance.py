import cv2
import numpy as np

def enhance_image(target):
    # Изменение размера изображения
    resized_image = cv2.resize(target, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Применение фильтра размытия Гаусса
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Увеличение резкости с использованием оператора Лапласа
    sharpened_image = cv2.Laplacian(blurred_image, cv2.CV_8U)

    # Разделение оттенков серого на отдельные каналы
    b, g, r = cv2.split(sharpened_image)

    # Применение гистограммного выравнивания к каждому каналу
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Объединение каналов обратно в цветное изображение
    equalized_image = cv2.merge((b_eq, g_eq, r_eq))

    return equalized_image

# Загрузка фотографии target
target = cv2.imread('found/sticker_0.jpg')

# Улучшение качества фотографии
enhanced_image = enhance_image(target)

# Отображение и сохранение улучшенного изображения
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)