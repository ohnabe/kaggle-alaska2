import torch
import numpy
import pandas as pd
from utils import utils
from config import Config
import argparse
import os
from torch.utils.data import Dataset
import glob
from transforms import transforms


class ALASKA2TestDataset(Dataset):
    def __init__(self, root, transforms=None, TTA=False):
        self.root = root
        self.img_list = []
        self.transforms = transforms
        self.TTA = TTA

        self.img_list = glob.glob(os.path.join(self.root, 'Test', '*.jpg'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img = utils.read_image(self.img_list[i])
        if self.TTA == 'hflip':
            img = img[:, ::-1, :].copy()
        elif self.TTA == 'vflip':
            img = img[::-1, :, :].copy()
        img = self.transforms(image=img)['image']
        #img /= 255.0
        return os.path.basename(self.img_list[i]), img



def main():

    parser = argparse.ArgumentParser(description='predict alaska2')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--submission_file', type=str)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--TTA', action='store_true')
    args = parser.parse_args()

    # get config
    config = Config()

    # create model
    model = utils.create_model(config)

    #load model
    state = torch.load(os.path.join(args.result_path, 'best_snapshot'))
    #print(type(state['models']['main']))
    model.load_state_dict(state['models']['main'])

    device = 'cuda'
    model.cuda()
    model.eval()

    # predict transform
    pred_trans = transforms.eval_transform(resize=(config.input_size_h, config.input_size_w), normalize=config.normalize)

    tta = [False]
    if args.TTA:
        tta.extend(['hflip','vflip'])

    for t in tta:
        # create dataset
        pred_dataset = ALASKA2TestDataset(config.data, pred_trans, TTA=t)

        # create data loader
        pred_loader = torch.utils.data.DataLoader(
                pred_dataset,
                batch_size = args.batchsize,
                num_workers = args.num_workers,
                shuffle = False,
            pin_memory = True
            )

        result = {'Id':[], 'Label':[]}
        for img_names, imgs in pred_loader:
            pred = model(imgs.cuda())
            pred = 1 - torch.nn.functional.softmax(pred, dim=1).data.cpu().numpy()[:, 0]

            result['Id'].extend(img_names)
            result['Label'].extend(pred)

        if t == 'hflip' or t == 'vflip':
            submission = pd.merge(submission, pd.DataFrame(result), how='inner', on='Id')
        else:
            submission = pd.DataFrame(result)

    if args.TTA:
        print(submission.columns)
        submission['Label'] = (submission['Label_x'] + submission['Label_y'] + submission['Label']) / 3
        submission = submission[['Id', 'Label']]

    submission.to_csv(os.path.join(args.result_path, args.submission_file), index=False)

if __name__ == '__main__':
    main()