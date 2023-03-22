import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage
import matplotlib as mpl
from torch.fft import fft2, ifft2, fftshift
from tqdm.notebook import tqdm
import scipy

### Simulating an equilateral triangle probe propagated to 0.4 mm (according to the parameters in https://opg.optica.org/directpdfaccess/896bd70c-57e2-4fcd-b8ee9f837d8ac6a2_446368/oe-29-2-1441.pdf?da=1&id=446368&seq=0&mobile=no

def get_triangle_mask():
    cx = cy = 0
    r = 1
    
    a3 = np.array([cy + r, cx]) * 3e-6
    
    ang1 = np.pi * 2 /3
    ang2 = ang1 * 2

    t1 = np.cos(ang1)
    t2 = np.sin(ang1)
    t3 = np.cos(ang2)
    t4 = np.sin(ang2)
    
    a2 = np.array([a3[0] * t1 - a3[1] * t2, a3[0] * t2 + a3[1] * t1])
    a1 = np.array([a3[0] * t3 - a3[1] * t4, a3[0] * t4 + a3[1] * t3])
    
    coords = np.array([a1, a2, a3])
    
    poly_path = mpl.path.Path(coords)
    
    x = np.linspace(-5e-6,5e-6,512)
    xx, yy = np.meshgrid(x, x)
    points = np.vstack((xx.flatten(),yy.flatten())).T
    
    mask = poly_path.contains_points(points).reshape(512, 512)
    return mask

