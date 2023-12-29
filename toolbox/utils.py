import numpy as np
import torch
from tqdm import tqdm
import os
import math
import random
import time
import torch.backends.cudnn as cudnn



class ClassWeight(object):

    def __init__(self, method):
        assert method in ['no', 'enet', 'median_freq_balancing']
        self.method = method

    def get_weight(self, dataloader, num_classes):
        if self.method == 'no':
            return np.ones(num_classes)
        if self.method == 'enet':
            return self._enet_weighing(dataloader, num_classes)
        if self.method == 'median_freq_balancing':
            return self._median_freq_balancing(dataloader, num_classes)

    def _enet_weighing(self, dataloader, num_classes, c=1.02):
        """Computes class weights as described in the ENet paper:

            w_class = 1 / (ln(c + p_class)),

        where c is usually 1.02 and p_class is the propensity score of that
        class:

            propensity_score = freq_class / total_pixels.

        References: https://arxiv.org/abs/1606.02147

        Keyword arguments:
        - dataloader (``data.Dataloader``): A data loader to iterate over the
        dataset.
        - num_classes (``int``): The number of classes.
        - c (``int``, optional): AN additional hyper-parameter which restricts
        the interval of values for the weights. Default: 1.02.

        """
        print('computing class weight .......................')
        class_count = 0
        total = 0
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = sample['label']
            label = label.cpu().numpy()

            # Flatten label
            flat_label = label.flatten()

            # Sum up the number of pixels of each class and the total pixel
            # counts for each label
            class_count += np.bincount(flat_label, minlength=num_classes)
            total += flat_label.size

        # Compute propensity score and then the weights for each class
        propensity_score = class_count / total
        class_weights = 1 / (np.log(c + propensity_score))

        return class_weights

    def _median_freq_balancing(self, dataloader, num_classes):
        """Computes class weights using median frequency balancing as described
        in https://arxiv.org/abs/1411.4734:

            w_class = median_freq / freq_class,

        where freq_class is the number of pixels of a given class divided by
        the total number of pixels in images where that class is present, and
        median_freq is the median of freq_class.

        Keyword arguments:
        - dataloader (``data.Dataloader``): A data loader to iterate over the
        dataset.
        whose weights are going to be computed.
        - num_classes (``int``): The number of classes

        """
        print('computing class weight .......................')
        class_count = 0
        total = 0
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = sample['label']
            label = label.cpu().numpy()

            # Flatten label
            flat_label = label.flatten()

            # Sum up the class frequencies
            bincount = np.bincount(flat_label, minlength=num_classes)

            # Create of mask of classes that exist in the label
            mask = bincount > 0
            # Multiply the mask by the pixel count. The resulting array has
            # one element for each class. The value is either 0 (if the class
            # does not exist in the label) or equal to the pixel count (if
            # the class exists in the label)
            total += mask * flat_label.size

            # Sum up the number of pixels found for each class
            class_count += bincount

        # Compute the frequency and its median
        freq = class_count / total
        med = np.median(freq)

        return med / freq


def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255.0 if normalized else cmap
    return cmap


def class_to_RGB(label, N, cmap=None, normalized=False):
    '''
        label: 2D numpy array with pixel-level classes shape=(h, w)
        N: number of classes, including background, should in [0, 255]
        cmap: list of colors for N class (include background) \
              if None, use VOC default color map.
        normalized: RGB in [0, 1] if True else [0, 255] if False

        :return 上色好的3D RGB numpy array shape=(h, w, 3)
    '''
    dtype = "float32" if normalized else "uint8"

    assert len(label.shape) == 2, f'label should be 2D, not {len(label.shape)}D'
    label_class = np.asarray(label)

    label_color = np.zeros((label.shape[0], label.shape[1], 3), dtype=dtype)

    if cmap is None:
        # 0表示背景为[0 0 0]黑色,1~N表示N个类别彩色
        cmap = color_map(N, normalized=normalized)
    else:
        cmap = np.asarray(cmap, dtype=dtype)
        cmap = cmap / 255.0 if normalized else cmap

    assert cmap.shape[0] == N, f'{N} classes and {cmap.shape[0]} colors not match.'

    # 给每个类别根据color_map上色
    for i_class in range(N):
        label_color[label_class == i_class] = cmap[i_class]

    return label_color


def tensor_classes_to_RGBs(label, N, cmap=None):
    '''used in tensorboard'''

    if cmap is None:
        cmap = color_map(N)
    else:
        cmap = np.asarray(cmap)

    label = label.clone().cpu().numpy()  # (batch_size, H, W)
    ctRGB = np.vectorize(lambda x: tuple(cmap[int(x)].tolist()))

    colored = np.asarray(ctRGB(label)).astype(np.float32)  # (batch_size, 3, H, W)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])

#
# def save_ckpt(logdir, ind, model):
#     state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
#     torch.save(state, os.path.join(logdir, ind+'model.pth'))
def save_ckpt(logdir, model):
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, os.path.join(logdir, 'model.pth'))

def load_ckpt(logdir, model):
    save_pth = os.path.join(logdir, 'model.pth')
    model.load_state_dict(torch.load(save_pth))
    return model


def compute_speed(model, input_size, device=0, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_weight_decay(model):

    import torch.nn as nn
    from torch.nn.modules.conv import _ConvNd
    from torch.nn.modules.batchnorm import _BatchNorm

    decays = []
    no_decays = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            decays.append(m.weight)
            if m.bias is not None:
                no_decays.append(m.bias)
        elif isinstance(m, _ConvNd):
            decays.append(m.weight)
            if m.bias is not None:
                no_decays.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.weight is not None:
                no_decays.append(m.weight)
            if m.bias is not None:
                no_decays.append(m.bias)

    assert len(list(model.parameters())) == len(decays) + len(no_decays)
    groups = [dict(params=decays), dict(params=no_decays, weight_decay=0.0)]
    return groups


if __name__ == '__main__':
    a = torch.randint(0, 10, (2, 50, 50))
    out = tensor_classes_to_RGBs(a, 10)
    print(out.shape)

    # print(out)
