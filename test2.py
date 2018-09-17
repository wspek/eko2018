import os
from PIL import Image

if __name__ == '__main__':
    path = '/home/wspek/Code/dev/eko2018/decompiled/'
    file_list = os.listdir(path)

    sum = 0
    for file_path in file_list:
        file_path = path + file_path
        imageObject = Image.open(file_path).convert("L")
        histogram = imageObject.histogram()
        sum = sum + histogram[255]

        test = 0

    print sum