import cv2
import os
import numpy as np

# Путь к папке со стикерами и к папке с обрезанными стикерами
input_folder = 'stickers_png'
# Укажите путь к исходной папке с PNG изображениями
output_folder = 'stickers_modifed'

def trim_images_in_folder(input_folder, output_folder):
    # Создаем выходную папку, если она еще не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # Обрабатываем только файлы с расширением PNG
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Загрузка изображения с альфа-каналом
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # Проверка наличия альфа-канала
            if image.shape[2] == 4:
                alpha_channel = image[:, :, 3]
                _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                print(f"Изображение {filename} не содержит альфа-канала.")
                continue

            if contours:
                # Объединение всех контуров и нахождение ограничивающего прямоугольника
                all_contours = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_contours)

                # Обрезка изображения с учетом альфа-канала
                cropped_image = image[y:y + h, x:x + w]
                cv2.imwrite(output_path, cropped_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            else:
                print(f"Контур для обрезки не найден в изображении {filename}.")

# Вызов функции
# trim_images_in_folder(input_folder, output_folder)

# all jpg images to png to save transparency

# def convert_images(input_folder, output_folder):
#     # Создаем папку для выходных изображений, если она не существует
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith('.jpg'):
#             file_path = os.path.join(input_folder, filename)
#             img = Image.open(file_path)
#             # Сохраняем изображение в формате PNG в папку назначения
#             img.save(os.path.join(output_folder, filename.replace('.jpg', '.png')), 'PNG')
#
# # Замените 'input_folder_path' на путь к папке с вашими JPG изображениями
# # и 'output_folder_path' на путь, куда вы хотите сохранить PNG изображения
# convert_images('stickers', 'stickers_png')