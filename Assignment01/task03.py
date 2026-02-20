import cv2
import numpy as np

img = cv2.imread("noisy_image.jpg", cv2.IMREAD_GRAYSCALE)

# Simple convolution function
def convolve(img, kernel):
    kernel = np.flip(kernel)  
    h, w = img.shape
    kh, kw = kernel.shape
    
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    
    out = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * kernel)
            
    return out

# 7x7 Gaussian kernel (sigma = 1)
size = 7
sigma = 1
ax = np.arange(-(size//2), size//2+1)
xx, yy = np.meshgrid(ax, ax)
gaussian = np.exp(-(xx**2+yy**2)/(2*sigma**2))
gaussian /= np.sum(gaussian)

# Denoise image
denoised = convolve(img, gaussian)
denoised = np.clip(denoised, 0, 255).astype(np.uint8)

# Sharpening kernel
sharpening_kernel = np.array([
    [1,4,6,4,1],
    [4,16,24,16,4],
    [6,24,-476,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1]
]) * (-1.0/256.0)

# Sharpen image
sharpened = convolve(denoised, sharpening_kernel)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Save results
cv2.imwrite("denoised.jpg", denoised)
cv2.imwrite("sharpened.jpg", sharpened)

print("done ")