#!/usr/bin/env python3
"""
Copyright 2020, Zixin Luo, HKUST.
Image matching example.
"""
import yaml
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from utils.opencvhelper import MatcherWrapper

from models import get_model


# tf.compat.v1.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def load_imgs(img_paths, max_dim):
    rgb_list = []
    gray_list = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
    return rgb_list, gray_list

def extract_local_features(gray_list, model_path, config):
    model = get_model('feat_model')(model_path, **config)
    descs = []
    kpts = []
    for gray_img in gray_list:
        desc, kpt, _ = model.run_test_data(gray_img)
        print('feature_num', kpt.shape[0])
        descs.append(desc)
        kpts.append(kpt)
    return descs, kpts


