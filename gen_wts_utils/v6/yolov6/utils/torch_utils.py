import torch.nn as nn
import os
from yolov6.utils.events import LOGGER
import requests

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def download_ckpt(path):
    """Download checkpoints of the pretrained models"""
    basename = os.path.basename(path)
    dir = os.path.abspath(os.path.dirname(path))
    os.makedirs(dir, exist_ok=True)
    LOGGER.info(f"checkpoint {basename} not exist, try to downloaded it from github.")
    # need to update the link with every release
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{basename}"
    r = requests.get(url, allow_redirects=True)
    assert r.status_code == 200, "Unable to download checkpoints, manually download it"
    open(path, 'wb').write(r.content)
    LOGGER.info(f"checkpoint {basename} downloaded and saved")