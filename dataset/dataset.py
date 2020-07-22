from torch.utils.data import Dataset
import numpy as np
import glob
import os
from utils import utils
import pandas as pd
import random


class Alaska2Dataset(Dataset):
    def __init__(self, root, transforms=None, train=True, k_fold=5, uniform=False, batchsize=16):
        self.root = root
        img_list = []
        label_list = []
        self.transforms = transforms
        self.train = train
        self.npy = False
        self.uniform = uniform
        self.batch_num = batchsize

        ext = '*.jpg'
        if os.path.exists(os.path.join(root, 'npy')):
            root = os.path.join(root, 'npy')
            ext = '*.npy'
            self.npy = True

        dir_list = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
        if os.path.exists(os.path.join(root, 'annotate.csv')):
            df = pd.read_csv(os.path.join(root, 'annotate.csv'))
        else:
            for i, dir in enumerate(dir_list):
                imgs = glob.glob(os.path.join(root, dir, ext))
                img_list.extend(imgs)
                label_list.extend([i] * len(imgs))
            df = pd.DataFrame(data={'img':img_list, 'label':label_list})

            df['img_file'] = df['img'].apply(lambda x: os.path.basename(x))
            img_df = df[['img_file']].drop_duplicates('img_file').reset_index()
            img_df['ind'] = img_df.index.values
            img_df['fold'] = img_df['ind'].apply(lambda x: x % k_fold)
            df = pd.merge(df, img_df[['img_file', 'fold']], how='inner', on='img_file')
            df = df.sample(frac=1, random_state=0).reset_index().drop(columns=['index'])
            df.to_csv(os.path.join(root, 'annotate.csv'))

        if train:
            self.df = df[df['fold'] > 0].reset_index()
            if self.uniform:
                self.img_name_uniq = [os.path.splitext(os.path.basename(x))[0] for x in list(self.df['img_file'].values)]
                self.img_name_uniq = list(set(self.img_name_uniq))
                self.selected = []
                self.batch = []
                self.img_dict = self.create_img_dict()

        else:
            self.df = df[df['fold'] == 0].reset_index()
        print(len(self.df))


    def __len__(self):
        return len(self.df)


    def __getitem__(self, i):
        if not self.uniform:
            if self.npy:
                img = np.load(self.df.loc[i, 'img']).astype(np.float32)
            else:
                img = utils.read_image(self.df.loc[i, 'img'])
            label = self.df.loc[i, 'label']
        else:
            if len(self.batch) == 0:
                self.create_batch()
            img_path, label = self.batch.pop(0)
            if self.npy:
                img = np.load(img_path).astype(np.float32)
            else:
                img = utils.read_image(img_path)
        img = self.transforms(image=img)['image']
        return img, label


    def create_batch(self):
        self.batch = []
        if len(self.selected) == len(self.img_name_uniq):
            self.selected = []

        target = list(set(self.img_name_uniq) - set(self.selected))
        random.shuffle(target)
        target = target[:self.batch_num // 4]
        self.selected = self.selected + target
        for i in range(len(target)):
            self.batch.extend(self.img_dict[target[i]])
        random.shuffle(self.batch)



    def create_img_dict(self):
        img_dict = {}
        for img_file, label in zip(list(self.df['img'].values), list(self.df['label'].values)):
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            if not img_name in img_dict:
                img_dict[img_name] = [(img_file, label)]
            else:
                img_dict[img_name].append((img_file, label))
        return img_dict

