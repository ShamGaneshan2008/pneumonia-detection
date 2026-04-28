import torch
import numpy as np
import cv2
from fastai.vision.all import *


def generate_gradcam(learn, img_path):
    img = PILImage.create(img_path)
    x, = learn.dls.test_dl([img]).one_batch()

    x.requires_grad_()

    # Run a forward pass and backprop on the top predicted class
    preds = learn.model(x)
    class_idx = preds.argmax(dim=1)
    preds[0, class_idx].backward()

    # Average the gradients across channels to get a single heatmap
    gradients = x.grad[0].cpu().numpy()
    heatmap = np.mean(gradients, axis=0)

    # Discard negative values — we only care about features that activate the class
    heatmap = np.maximum(heatmap, 0)

    # 1e-8 prevents division by zero if the heatmap is all zeros
    heatmap /= heatmap.max() + 1e-8

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    return heatmap