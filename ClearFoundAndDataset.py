import os
import shutil

a = "found"
b = "dataset"

def clear_folder(folder_path):
    # Проверка существования папки
    if os.path.exists(folder_path):
        # Удаление содержимого папки
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Папка {folder_path} успешно очищена.")
    else:
        print(f"Папка {folder_path} не существует.")

clear_folder(a)
clear_folder(b)