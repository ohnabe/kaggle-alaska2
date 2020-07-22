import multiprocessing
from tqdm import tqdm
import cv2
import os
import numpy as np
import glob

ROOT='/media/realnabe/0b60e9b4-6f7e-4a46-bf2a-be31d1ccbf25/Kaggle/ALASKA2/data'

def create_npy(img_file):
    kind = img_file.split('/')[-2]
    npy_path = os.path.join(ROOT, 'npy', kind, os.path.splitext(os.path.basename(img_file))[0] + '.npy')
    dir_name = os.path.dirname(npy_path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    np.save(npy_path, img)


def main():
    img_list = glob.glob(os.path.join(ROOT, '*', '*.jpg'))
    print(len(img_list))

    p = multiprocessing.Pool(8)
    p.map(create_npy, img_list)
    p.close()

    #for i in tqdm(range(len(img_list))):
    #    create_npy(img_list[i])


if __name__ == '__main__':
    main()