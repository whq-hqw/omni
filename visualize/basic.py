"""
# Copyright (c) 2018 Works Applications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
import os, math, time, warnings, datetime
import torch
import cv2
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import omni_torch.utils as util
import matplotlib.pyplot as plt
from tkinter import _tkinter
try:
    import scipy.signal as ss
except ImportError:
    pass

def visualize_gradient(args, net, img_ratio=16/9, minimun_rank=2):
    for name, param in net.named_parameters():
        if param.grad is None:
            # If the grad of parameter does not exist
            # or current layer does not need BP
            continue
        if len(param.shape) < minimun_rank:
            continue
        if len(param.shape) == 2:
            param = param.unsqueeze(0)
        if len(param.shape) == 3:
            param = param.unsqueeze(0)
        if len(param.shape) > 4:
            # We do not visualize tensor with such shape
            continue
        norm = float(param.grad.norm(2))
        norm_grad = param.grad.data / norm
        min_v = torch.min(norm_grad)
        diff = torch.max(norm_grad) - min_v
        if diff != 0:
            norm_grad = (norm_grad - min_v) / diff * 255
        title = "gradient l-2 norm: " + str(norm)[:8]
        if not os.path.exists(os.path.join(args.grad_log, name)):
            os.mkdir(os.path.join(args.grad_log, name))
        img_path = os.path.join(args.grad_log, name, name + "_" + str(args.curr_epoch).zfill(4) + ".jpg")
        plot_tensor(args, norm_grad, title=title, path=img_path, ratio=img_ratio, sub_margin=1)


def to_image(args, tensor, margin, deNormalize, sub_title=None, font_size=0.5):
    if args is None:
        args = edict({
            "img_mean": (0.5, 0.5, 0.5),
            "img_std": (1.0, 1.0, 1.0),
            "img_bias": (0.0, 0.0, 0.0),
            "img_bit": 8,
        })
    assert len(tensor.shape) == 3
    array = tensor.data.to("cpu").numpy()
    array = array.transpose((1, 2, 0))
    if deNormalize:
        array = util.denormalize_image(args, array)
        #array = array.astype("uint8")
    num = array.shape[2]
    if num in [1, 3]:
        # if array depth is 1 or 3 than this is an grayscale or RGB image
        canvas = array
    else:
        v = array.shape[0]
        h = array.shape[1]
        v_num = int(max(math.sqrt(num), 1))
        h_num = int(math.ceil(num / v_num))
        # Create a white canvas contains margin for each patches
        canvas = np.zeros((v_num * v + (v_num - 1) * margin, h_num * h + (h_num - 1) * margin)) + 255
        for i in range(v_num):
            for j in range(h_num):
                if (i * h_num + j) >= num:
                    break
                canvas[i * v + i * margin: (i + 1) * v + i * margin,
                j * h + j * margin: (j + 1) * h + j * margin] = array[:, :, i * h_num + j]
    if len(canvas.shape) == 2:
        canvas = np.expand_dims(canvas, axis=-1)
    if canvas.shape[2] == 1:
        canvas = np.tile(canvas, (1, 1, 3))
    height, width = canvas.shape[0], canvas.shape[1]
    w_complement = 0
    if sub_title is not None:
        sub_title = str(sub_title)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_w, text_h = cv2.getTextSize(sub_title, font, 1, 2)[0]
        text_w = int(text_w * font_size)
        text_h = int(text_h * font_size)
        if int(text_w * 1.1) > width:
            # Because we want to give a little margin on the start and end of text
            w_complement = int((text_w * 1.1 - width) / 2)
            width = int(text_w * 1.1)
        h_complement = text_h * 2
        height += h_complement
        painter = np.zeros((height, width, 3)) + 255
        position = (int((width - text_w) / 2), int(text_h * 1.5))
        cv2.putText(painter, sub_title, position, font, fontScale=font_size, color=(48, 33, 255), thickness=1)
        painter[h_complement:, w_complement:w_complement+canvas.shape[1], :] = canvas
        canvas = painter
    return canvas


def plot_tensor(args, tensor, path=None, title=None, sub_title=None, op=to_image, ratio=1,
                margin=5, sub_margin=True, deNormalize=True, font_size=1, bg_color=255):
    """
    This is a function to plot one tensor at a time.
    :param tensor: can be gradient, parameter, data_batch, etc.
    :param op: operation that convert a (C, W, H) tensor into nd-array
    :param title:
    :param ratio:
    :param path: if not None, function will save the image onto the local disk.
    :param margin: the distance between each image patches
    :return:
    """
    num = tensor.size(0)
    if sub_title:
        assert num == len(sub_title), "Number of sub titles must be same as the number of plot tensors"
        sub_length = len(sub_title[0])
        for sub in sub_title:
            if len(str(sub)) != sub_length:
                warnings.warn("length of string in sub_title are recommended to be same.")
                break
    else:
        sub_title = [None] * num
    v = int(max(math.sqrt(num / ratio), 1))
    h = int(math.ceil(num / v))
    if sub_margin:
        if type(sub_margin) is bool:
            patch_margin = int(margin / 2)
        elif type(sub_margin) is int:
            patch_margin = sub_margin
    else:
        patch_margin = 0

    # Find out the size of each small image patches
    img = op(args, tensor[0], patch_margin, deNormalize, sub_title[0])
    #img = plot_tensor(args,tensor[0], title=sub_title[0], ratio=ratio, margin=sub_margin, deNormalize=deNormalize,
                      #font_size=font_size/2, bg_color=bg_color)
    v_p, h_p, c_p = img.shape
    # Create a large white canvas to plot the small patches
    height = v * v_p + (v + 1) * margin
    width = h * h_p + (h + 1) * margin
    w_complement = 0
    h_complement = 0
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_w, text_h = cv2.getTextSize(title, font, 1, 2)[0]
        text_w = int(text_w * font_size)
        text_h = int(text_h * font_size)
        if int(text_w * 1.1) > width:
            # Because we want to give a little margin on the start and end of text
            w_complement = int((text_w * 1.1 - width) / 2)
            width = int(text_w * 1.1)
        h_complement = text_h * 2
        height += h_complement
        canvas = np.zeros((height, width, 3)) + bg_color
        position = (int((width - text_w) / 2), int(text_h * 1.5))
        cv2.putText(canvas, title, position, font, fontScale=font_size, color=(48, 33, 255), thickness=2)
        # cv2.imwrite(path, canvas)
    else:
        canvas = np.zeros((height, width, 3)) + bg_color

    # fig, axis = plt.subplots(ncols=h_num, nrows=v_num, figsize = (h_sub, v_sub))
    if v == h == 1:
        canvas[h_complement + margin:h_complement + margin + img.shape[0],
                        margin:margin + img.shape[1], :] = img
        if path:
            cv2.imwrite(path, canvas)
        else:
            return canvas
    else:
        for i in range(v):
            for j in range(h):
                if (i * h + j) >= num:
                    break
                h_s = i * v_p + (i + 1) * margin + h_complement
                h_e = (i + 1) * (margin + v_p) + h_complement
                w_s = j * h_p + (j + 1) * margin + w_complement
                w_e = (j + 1) * (margin + h_p) + w_complement
                try:
                    canvas[h_s: h_e, w_s: w_e, :] = op(args, tensor[i * h + j], patch_margin, deNormalize, sub_title[i * h + j])
                except ValueError:
                    canvas[h_s: h_e, w_s: w_e, :] = cv2.resize(op(args, tensor[i * h + j], patch_margin, deNormalize, sub_title[i * h + j]),
                                                               (w_e-w_s, h_e-h_s))
    if path:
        cv2.imwrite(path, canvas)
    else:
        return canvas


def plot_curves(line_data, line_labels, save_path=None, name=None, epoch=None,
                window=5, fig_size=(18, 6), bound=None, grid=True, ax=None, title=None):
    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=fig_size)
        save_fig = True
    t = np.asarray(list(range(len(line_data[0]))))
    if ss and window > 1 and window % 2 == 1:
        line_data = [ss.savgol_filter(_, window, 2) for _ in line_data]
    for i, loss in enumerate(line_data):
        ax.plot(t, loss, label=line_labels[i])
    ax.legend(loc='upper right')
    ax.set_xlabel('epoch')
    ax.grid(grid)
    if title:
        ax.set_title(title)
    if bound is not None:
        ax.set_ylim(bound["low"], bound["high"])
    if save_fig and save_path:
        ep = "_" + str(int(epoch)).zfill(4) if epoch is not None else ""
        img_name = name + ep + ".jpg"
        plt.savefig(os.path.join(save_path, img_name))
        plt.close()


def plot_multi_loss_distribution(multi_line_data, multi_line_labels, save_path=None, name=None,
                                 epoch=None, window=5, fig_size=(18, 12), bound=None,
                                 grid=True, titles=None):
    assert len(multi_line_data) == len(multi_line_labels)
    if titles is None:
        titles = [None] * len(multi_line_data)
    if bound is None:
        bound = [None] * len(multi_line_data)
    assert len(multi_line_data) == len(titles) == len(bound)
    fig, axes = plt.subplots(len(multi_line_data), 1, figsize=fig_size)
    for i, losses in enumerate(multi_line_data):
        plot_curves(losses, multi_line_labels[i], window=window,
                    bound=bound[i], grid=grid, ax=axes[i], title=titles[i])
    if save_path:
        ep = "_" + str(int(epoch)).zfill(4) if epoch is not None else ""
        img_name = name + ep + ".jpg"
        plt.savefig(os.path.join(save_path, img_name))
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    pass