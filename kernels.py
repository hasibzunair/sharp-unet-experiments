from scipy.ndimage import gaussian_filter
from skimage.filters import unsharp_mask
import numpy as np

def gauss_2D(shape=(3,3),sigma=5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



def lpf(tensor, sigma=5):
    
    "A Gaussian filter"
    tensor = gaussian_filter(tensor, sigma=sigma)
    return tensor



def sharpen(tensor, sigma=5):
    
    """
    Enhance via HPF
    
    sigma: Standard deviation for Gaussian kernel. The standard deviations
    of the Gaussian filter are given for each axis as a sequence
    , or as a single number, in which case it is equal for all axes.
    
    """
    
    # blur
    tensor_b1 = gaussian_filter(tensor, sigma=sigma)
    
    alpha = 30
    #Formula: enhanced image = original + alpha * (original - blurred)
    tensor_sharpened = tensor + alpha * (tensor - tensor_b1)
    
    return tensor_sharpened




def unsharp_mask_filter(tensor, radius=5, amount=1):
    """
    Apply unsharp mask
    Formula: enhanced image = original + amount * (original - blurred)

    HPF: https://diffractionlimited.com/help/maximdl/High-Pass_Filtering.htm
    Unsharpmask: https://diffractionlimited.com/help/maximdl/Unsharp_Mask_Basics.htm
    Doc: https://scikit-image.org/docs/dev/auto_examples/filters/plot_unsharp_mask.html
    
    Args:
    
    tensor: tensor to process
    radius: The radius parameter in the unsharp masking filter refers to the sigma
    parameter of the gaussian filter.
    
    amount: The details will be amplified with this factor. The factor could be 0
    or negative. Typically, it is a small positive number, e.g. 1.0.
    
    """
    # Try 1,1 5,2 20,1
    tensor = unsharp_mask(tensor, radius=radius, amount=amount)
    return tensor
