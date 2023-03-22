import torch
import matplotlib.pyplot as plt
from wavefront import Wavefront

def plane2_forward_model(obj_guess: torch.Tensor, probe_plane2: Wavefront):
    exit_wave = probe_plane2.obj_mult(obj_guess)
    farfield = exit_wave.prop_farfield()
    return farfield


def plane1_forward_model(probe_plane1_guess: Wavefront, transfer_function=None):
    nearfield = probe_plane1_guess.prop_nearfield(prop_dist=prop_dist, wavelength=wavelength, pixel_size=pixel_size, transfer_function=transfer_function)
    return nearfield

def plane1_constraint(probe: Wavefront, mask_tensor):
    return probe.obj_mult(mask_tensor)

def loss_fn(guess: torch.Tensor, true: torch.Tensor):
    return torch.mean(torch.abs(guess - true)**2)


def amplitude_constraint(obj, max_val=1.0, epsilon=1e-8):
    obj_ampl = torch.abs(obj)
    new_ampl = torch.clamp(obj_ampl, min=0, max=max_val)
    scaling = new_ampl / (obj_ampl + epsilon)
    return scaling.type(torch.complex64) * obj

def plot2(wavefront, titles=[None, None], suptitle=None):
    fig, axs = plt.subplots(1,2,figsize=[8,4], constrained_layout=True)
    plt.subplot(1,2,1)
    plt.imshow(wavefront.fftshift().amplitude_numpy)
    plt.colorbar()
    title0 = titles[0] if titles[0] is not None else 'Ampl'
    plt.title(title0)

    plt.subplot(1,2,2)
    plt.imshow(wavefront.fftshift().phase_numpy)
    plt.colorbar()
    title1 = titles[1] if titles[1] is not None else 'Phase'
    plt.title(title1)
    
    if suptitle is not None:
        plt.suptitle(suptitle)
        
    plt.show()
