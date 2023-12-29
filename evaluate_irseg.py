import os
import time
from tqdm import tqdm
from PIL import Image
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_dataset
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox.utils import class_to_RGB, load_ckpt

from toolbox.datasets.irseg import IRSeg

def evaluate(logdir):
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')
    testset = IRSeg(cfg, mode='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])
   
    def load_ckpt(logdir, model):

        save_pth = os.path.join(logdir, 'model.pth')
        _dict = torch.load(save_pth)
        model_dict = model.state_dict()
        _dict = {k: v for k, v in _dict.items() if k in model_dict}
        model_dict.update(_dict)
        model.load_state_dict(model_dict)
        return model
    model = get_model(cfg).to(device)
    model = load_ckpt(logdir, model)

    running_metrics_val = runningScore(cfg['n_classes'])#, ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    with torch.no_grad():
        model.eval()
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):

            time_start = time.time()
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            out = model(image, depth)
   

            predict = out.max(1)[1].cpu().numpy()  # [1, h, w]
            label = label.cpu().numpy()
            running_metrics_val.update(label, predict)

            time_meter.update(time.time() - time_start, n=image.size(0))



    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    print('class Acc:')
    for k, v in metrics[1].items():
        print(k, v)
    print('class IoU:')
    for k, v in metrics[2].items():
        print(k, v)
    print('inference time per image: ', time_meter.avg)
    print('inference fps: ', 1 / time_meter.avg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", default="./checkpoint/ieseg", type=str, help="run logdir")
    parser.add_argument("-s", type=bool, default="./checkpoint/ieseg", help="save predict or not")

    args = parser.parse_args()

    evaluate(args.logdir, save_predict=args.s)
