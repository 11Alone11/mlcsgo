from tkinter import filedialog

import cv2
import os
import math
import numpy as np
from shapely.geometry import box
import matplotlib.pyplot as plt
import glob
import pyperclip
import tkinter as tk
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
        image_target = cv2.imread(f"{Result_input_path}/{sticker_found}")
        j = 0
        for sticker_base in sticker_list:
            sticker_path = os.path.join(stickers_data, sticker_base)
            sticker_name = os.path.basename(sticker_path)
            sticker = cv2.imread(sticker_path)
            sticker = ScalePicture(sticker).scaleShacal()
            a, b = ScalePicture(image_target, sticker).scaleBoth()
            # Преобразование изображений из BGR в RGB (для отображения с помощью matplotlib)
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            # Вычисление гистограмм цветовых каналов для каждого изображения
            histogram1 = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            histogram2 = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            # Нормализация гистограмм
            cv2.normalize(histogram1, histogram1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(histogram2, histogram2, 0, 1, cv2.NORM_MINMAX)
            # Вычисление коэффициента корреляции Хистограмм
            probability[i][j] = cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_CORREL)
            sticker_names[i][j] = sticker_name
            j += 1
        i += 1

    return probability, sticker_names

def saver_drawer(probability, sticker_names):
    i = 0
    for sticker_found in os.listdir(Result_input_path):
        image_target = cv2.imread(f"{Result_input_path}/{sticker_found}")
        max_index = np.argmax(probability[i])
        max_prob = probability[i][max_index]
        sticker_name = sticker_names[i][max_index]
        cv2.imwrite(f"finally/{i}_{sticker_name}_{max_prob}.jpg", image_target)
        image_target = ScalePicture(image_target).scaleFirst_delete()
        #cv2.imshow(f"{i}_{sticker_name}_{max_prob}", image_target)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        i += 1

class ScalePicture:
    def __init__(self, img_1, img_2=None):
        self.img_1 = img_1
        self.img_2 = img_2
        self.scaled_img_1 = None
        self.scaled_img_2 = None

    def scaleFirst_delete(self):
        scale_percent = 400  # Процент от исходного размера
        width = int(self.img_1.shape[1] * scale_percent / 100)
        height = int(self.img_1.shape[0] * scale_percent / 100)
        dim = (width, height)
        scaled_img_1 = cv2.resize(self.img_1, dim, interpolation=cv2.INTER_AREA)
        self.scaled_img_1 = scaled_img_1
        return self.scaled_img_1

    def scaleShacal(self):
        scale_percent = 5  # Процент от исходного размера
        width = int(self.img_1.shape[1] * scale_percent / 100)
        height = int(self.img_1.shape[0] * scale_percent / 100)
        dim = (width, height)
        scaled_img_1 = cv2.resize(self.img_1, dim, interpolation=cv2.INTER_AREA)
        self.scaled_img_1 = scaled_img_1
        scale_percent = 1000 / scale_percent  # Процент от исходного размера
        width = int(self.scaled_img_1.shape[1] * scale_percent / 100)
        height = int(self.scaled_img_1.shape[0] * scale_percent / 100)
        dim = (width, height)
        scaled_img_1 = cv2.resize(self.scaled_img_1, dim, interpolation=cv2.INTER_AREA)
        self.scaled_img_1 = scaled_img_1
        return self.scaled_img_1
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
original_img = None
def open_image():
    global original_img
    # Открытие диалогового окна для выбора файла или вставки изображения из буфера обмена
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])

    # Если файл не был выбран, попытка получить данные из буфера обмена
    if not file_path:
        clipboard_data = pyperclip.paste()
        if clipboard_data.startswith('data:image'):
            file_path = 'clipboard_image.png'
            with open(file_path, 'wb') as f:
                f.write(pyperclip.paste().split(',')[1].decode('base64'))

    # Проверка, что файл был выбран или данные из буфера обмена были получены
    if file_path:
        # Проверка формата файла на изображение
        image_formats = ['png', 'jpg', 'jpeg']
        file_ext = file_path.lower().split('.')[-1]
        if file_ext in image_formats:
            # Загрузка изображения
            original_img = cv2.imread(file_path)
        else:
            result_label.config(text="Выбранный файл или данные из буфера обмена не являются изображением.")
    else:
        result_label.config(text="Файл или данные из буфера обмена не выбраны.")


# Создание графического интерфейса с кнопкой "Открыть изображение" и меткой для вывода результата
root = tk.Tk()
root.title("Открыть изображение")

button = tk.Button(root, text="Открыть изображение", command=open_image)
button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
#original_img = cv2.imread('Images/h4.jpg')
edited_img1 = cv2.imread('Images/h3.jpg')

original_img, edited_img1 = ScalePicture(original_img, edited_img1).scaleBoth()

# Изменение размера edited_img для соответствия размеру original_img
if np.any(original_img[0] != edited_img1[0]) and np.any(original_img[1] != edited_img1[1]):
    edited_img1 = cv2.resize(edited_img1, (original_img.shape[1], original_img.shape[0]))

# Нахождение различий
diff = cv2.absdiff(original_img, edited_img1)

alfa = ScalePicture(diff).scaleFirst()
#cv2.imshow("Difference", alfa)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

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
            and ((r[0] > 0.9 * r[1] or r[0] > 1.1 * r[1]) and (r[0] > 0.9 * math.sqrt(area(r)) or r[0] > 1.1 * math.sqrt(area(r)))):
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
##cv2.imwrite(f"{save_path}/detected_stickers.jpg", original_img)
#cv2.imshow("Detected", ScalePicture(original_img).scaleFirst())
#cv2.waitKey(0)

Result_input_path = 'found'
stickers_data = 'stickers_modifed'

a, b = correlate(Result_input_path, stickers_data)
saver_drawer(a, b)

# Путь к папке с изображениями
image_folder = 'finally/'

# Получение списка файлов изображений в папке
image_paths = glob.glob(image_folder + '*.jpg')

# Загрузка изображений и их названий
images = [cv2.imread(image_path) for image_path in image_paths]
image_names = [image_path.split('/')[-1] for image_path in image_paths]

# Определение размера изображений и создание фигуры с подходящим размером
num_images = len(images)
fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))

# Перебор изображений и их названий и их отображение
for i in range(num_images):
    axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    axes[i].set_title(image_names[i])
    axes[i].axis('off')

# Отображение скролящегося интерфейса с изображениями
plt.tight_layout()
plt.show()