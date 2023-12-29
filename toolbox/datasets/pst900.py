import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class PST900(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.45369372, 44.2457428, 31.66502391, 46.40709901, 30.13909209])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
        else:
            raise (f"{cfg['class_weight']} not support.")

        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()

        image = Image.open(os.path.join(self.root, 'rgb', image_path+'.png'))
        mask = Image.open(os.path.join(self.root, 'mask', image_path+'.png'))
        boundary = Image.open(os.path.join(self.root, 'boundary', image_path+'.png'))
        attention_map = Image.open(os.path.join(self.root, 'attention_map', image_path+'.png'))

        label = Image.open(os.path.join(self.root, 'labels', image_path+'.png'))
        depth = Image.open(os.path.join(self.root, 'thermal', image_path+'.png'))
        depth = depth.convert('RGB')

        sample = {
            'image': image,
            'depth': depth,
            'mask': mask,
            'boundary': boundary,
            'attention_map': attention_map,
            'label': label,
        }

        if self.mode in ['train', 'trainval'] and self.do_aug:
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['boundary'] = torch.from_numpy(np.asarray(sample['boundary'], dtype=np.int64) / 255.).long()
        sample['mask'] = torch.from_numpy(np.asarray(sample['mask'], dtype=np.int64) / 255.).long()
        sample['attention_map'] = torch.from_numpy(np.asarray(sample['attention_map'], dtype=np.int64) / 255.).long()

        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = image_path.strip().split('/')[-1]
        return sample

    @property
    def cmap(self):
        return [
            (0,0,0),         # background
            (0,0,255),      # fire extinguisher
            (0,255,0),       # backpack
            (255,0,0),     # hand drill
            (255,255,255),       # survivor

        ]
