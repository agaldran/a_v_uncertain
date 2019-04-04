import torch.nn.init as init
import torch.nn as nn

import numpy as np
import cv2

# TODO set the argument for diiferent options on initialization
def initialize_weights(method='kaiming', *models):
    for model in models:
        for module in model.modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                if method == 'kaiming':
                    init.kaiming_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'xavier':
                    init.xavier_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'orthogonal':
                    init.orthogonal(module.weight.data, np.sqrt(2.0))
                elif method == 'normal':
                    init.normal(module.weight.data,mean=0, std=0.02)
                if module.bias is not None:
                    init.constant(module.bias.data,0)



def morph_postprocess(img):
    kernel = np.ones((5,5),np.uint8)
    
    # --- opening ---
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # --- closing ---
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img
