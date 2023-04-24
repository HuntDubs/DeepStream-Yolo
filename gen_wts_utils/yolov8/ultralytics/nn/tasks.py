from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import contextlib

from ultralytics.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify,
                                    Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus,
                                    GhostBottleneck, GhostConv, Pose, Segment)
from utils.torch_utils import LOGGER, yaml_load
from utils.checks import check_yaml
from utils.torch_utils import ( initialize_weights, make_divisible ) 


class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x, profile=False, visualize=False):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.
        Args:
            x (torch.Tensor): The input image tensor
            profile (bool): Whether to profile the model, defaults to False
            visualize (bool): Whether to return the intermediate feature maps, defaults to False
        Returns:
            (torch.Tensor): The output of the network.
        """
        return self._forward_once(x, profile, visualize)

    # def _forward_once(self, x, profile=False, visualize=False):
    #     """
    #     Perform a forward pass through the network.
    #     Args:
    #         x (torch.Tensor): The input tensor to the model
    #         profile (bool):  Print the computation time of each layer if True, defaults to False.
    #         visualize (bool): Save the feature maps of the model if True, defaults to False
    #     Returns:
    #         (torch.Tensor): The last output of the model.
    #     """
    #     y, dt = [], []  # outputs
    #     for m in self.model:
    #         if m.f != -1:  # if not from previous layer
    #             x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    #         if profile:
    #             self._profile_one_layer(m, x, dt)
    #         x = m(x)  # run
    #         y.append(x if m.i in self.save else None)  # save output
    #         if visualize:
    #             LOGGER.info('visualize feature not yet supported')
    #             # TODO: feature_visualization(x, m.type, m.i, save_dir=visualize)
    #     return x

    # def _profile_one_layer(self, m, x, dt):
    #     """
    #     Profile the computation time and FLOPs of a single layer of the model on a given input.
    #     Appends the results to the provided list.
    #     Args:
    #         m (nn.Module): The layer to be profiled.
    #         x (torch.Tensor): The input data to the layer.
    #         dt (list): A list to store the computation time of the layer.
    #     Returns:
    #         None
    #     """
    #     c = m == self.model[-1]  # is final layer, copy input as inplace fix
    #     o = thop.profile(m, inputs=[x.clone() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
    #     t = time_sync()
    #     for _ in range(10):
    #         m(x.clone() if c else x)
    #     dt.append((time_sync() - t) * 100)
    #     if m == self.model[0]:
    #         LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
    #     LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
    #     if c:
    #         LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # def fuse(self, verbose=True):
    #     """
    #     Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
    #     computation efficiency.
    #     Returns:
    #         (nn.Module): The fused model is returned.
    #     """
    #     if not self.is_fused():
    #         for m in self.model.modules():
    #             if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
    #                 m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #                 delattr(m, 'bn')  # remove batchnorm
    #                 m.forward = m.forward_fuse  # update forward
    #             if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
    #                 m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
    #                 delattr(m, 'bn')  # remove batchnorm
    #                 m.forward = m.forward_fuse  # update forward
    #         self.info(verbose=verbose)

    #     return self

    # def is_fused(self, thresh=10):
    #     """
    #     Check if the model has less than a certain threshold of BatchNorm layers.
    #     Args:
    #         thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.
    #     Returns:
    #         (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
    #     """
    #     bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    #     return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    # def info(self, verbose=True, imgsz=640):
    #     """
    #     Prints model information
    #     Args:
    #         verbose (bool): if True, prints out the model information. Defaults to False
    #         imgsz (int): the size of the image that the model will be trained on. Defaults to 640
    #     """
    #     model_info(self, verbose=verbose, imgsz=imgsz)

    # def _apply(self, fn):
    #     """
    #     `_apply()` is a function that applies a function to all the tensors in the model that are not
    #     parameters or registered buffers
    #     Args:
    #         fn: the function to apply to the model
    #     Returns:
    #         A model that is a Detect() object.
    #     """
    #     self = super()._apply(fn)
    #     m = self.model[-1]  # Detect()
    #     if isinstance(m, (Detect, Segment)):
    #         m.stride = fn(m.stride)
    #         m.anchors = fn(m.anchors)
    #         m.strides = fn(m.strides)
    #     return self

    # def load(self, weights, verbose=True):
    #     """Load the weights into the model.
    #     Args:
    #         weights (dict) or (torch.nn.Module): The pre-trained weights to be loaded.
    #         verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
    #     """
    #     model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
    #     csd = model.float().state_dict()  # checkpoint state_dict as FP32
    #     csd = intersect_dicts(csd, self.state_dict())  # intersect
    #     self.load_state_dict(csd, strict=False)  # load
    #     if verbose:
    #         LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Run forward pass on input image(s) with optional augmentation and profiling."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # def _forward_augment(self, x):
    #     """Perform augmentations on input image x and return augmented inference and train outputs."""
    #     img_size = x.shape[-2:]  # height, width
    #     s = [1, 0.83, 0.67]  # scales
    #     f = [None, 3, None]  # flips (2-ud, 3-lr)
    #     y = []  # outputs
    #     for si, fi in zip(s, f):
    #         xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
    #         yi = self._forward_once(xi)[0]  # forward
    #         # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
    #         yi = self._descale_pred(yi, fi, si, img_size)
    #         y.append(yi)
    #     y = self._clip_augmented(y)  # clip augmented tails
    #     return torch.cat(y, -1), None  # augmented inference, train

    # @staticmethod
    # def _descale_pred(p, flips, scale, img_size, dim=1):
    #     """De-scale predictions following augmented inference (inverse operation)."""
    #     p[:, :4] /= scale  # de-scale
    #     x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
    #     if flips == 2:
    #         y = img_size[0] - y  # de-flip ud
    #     elif flips == 3:
    #         x = img_size[1] - x  # de-flip lr
    #     return torch.cat((x, y, wh, cls), dim)

    # def _clip_augmented(self, y):
    #     """Clip YOLOv5 augmented inference tails."""
    #     nl = self.model[-1].nl  # number of detection layers (P3-P5)
    #     g = sum(4 ** x for x in range(nl))  # grid points
    #     e = 1  # exclude layer count
    #     i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
    #     y[0] = y[0][..., :-i]  # large
    #     i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
    #     y[-1] = y[-1][..., i:]  # small
    #     return y

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        # if verbose:
        #     LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_stem(new_stem)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d

def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.
    Args:
        model_path (str) or (Path): The path to the YOLO model's YAML file.
    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''
