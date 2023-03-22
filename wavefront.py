import torch
import numpy as np

from torch.fft import fft2, ifft2, fftshift


def propFF(wavefront: torch.tensor,
    backward=False,
):
    
    if backward:
        new_wavefront = ifft2(wavefont)
    else:
        new_wavefront = fft2(wavefront)
    return new_wavefront

def propTF(
    wavefront: torch.tensor,
    wavelength: float,
    prop_dist: float,
    pixel_size: tuple[float, float],
    reuse_transfer_function: torch.tensor = None,
    backward: bool = False,
    cuda: bool = False,
) -> "Wavefront":
    """Propagation of the supplied wavefront using the Transfer Function function.

    This propagation method is also referred to as *angular spectrum* or *fresnel* propagation.

    Parameters
    ----------
    reuse_transfer_function : bool
        Reuse provided transfer function.
    transfer_function : array_like(complex)
        Transfer function after fftshift of reciprocal coordinates.
    prop_dist : float
        Propagation distance (in m).
    backward : bool
        Backward propagation.
    Returns
    -------
    wavefront : Wavefront
        Wavefront after propagation
    """

    nx = wavefront.shape[-1]
    ny = wavefront.shape[-2]


    if reuse_transfer_function is None:
        k = 2 * np.pi / wavelength

        # reciprocal space pixel size
        rdy = wavelength * prop_dist / (ny * pixel_size[0])
        rdx = wavelength * prop_dist / (nx * pixel_size[1])

        # reciprocal space coords
        x = torch.arange(-nx // 2, nx // 2) * rdx
        y = torch.arange(-ny // 2, ny // 2)[:, None] * rdy

        reuse_transfer_function = fftshift(torch.exp(-1j * k / (2 * prop_dist) * (x**2 + y**2)))

    if backward:
        new_wavefront = fft2(ifft2(wavefront) / reuse_transfer_function)
    else:
        new_wavefront = ifft2(reuse_transfer_function * fft2(wavefront))
    return new_wavefront, reuse_transfer_function

class Wavefront:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        
    def prop_farfield(self,  wavelength = None, prop_dist = None, backward=False):
        if prop_dist is not None:
            raise NotImplementedError
        new_tensor = propFF(self.tensor, backward=backward)
        return Wavefront(tensor=new_tensor)
    
    def prop_nearfield(self, prop_dist, wavelength, pixel_size, transfer_function=None, backward=False, return_transfer_function=False):

        new_tensor, transfer_function = propTF(self.tensor, wavelength, prop_dist, pixel_size, transfer_function, backward)
        new_wavefront = Wavefront(new_tensor)
        
        if return_transfer_function:
            return new_wavefront, transfer_function
        else:
            return new_wavefront
    
    def fftshift(self):
        return Wavefront(tensor=fftshift(self.tensor))
    
    def cuda(self):
        return Wavefront(tensor=self.tensor.cuda())
    
    def cpu(self):
        return Wavefront(tensor=self.tensor.cpu())
    
    @property
    def amplitude(self):
        return torch.abs(self.tensor)
    
    @property
    def phase(self):
        return torch.angle(self.tensor)
    
    @property
    def numpy(self):
        return np.array(self.tensor.cpu())
    
    @property
    def amplitude_numpy(self):
        return np.array(self.amplitude.cpu())
    
    @property
    def phase_numpy(self):
        return np.array(self.phase.cpu())
    
    def obj_mult(self, obj):
        new_tensor = self.tensor * fftshift(obj)
        return Wavefront(tensor=new_tensor)