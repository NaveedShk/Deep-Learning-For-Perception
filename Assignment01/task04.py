import cv2
import numpy as np


image = cv2.imread("shelf.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)

image = image.astype(np.float32)
template = template.astype(np.float32)


#mean subtraction 
image = image - np.mean(image)
template = template - np.mean(template)


#for convulution 
def convolve(image, kernel):
    kernel = np.flip(kernel)  
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(image, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    
    output = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i,j] = np.sum(region * kernel)
            
    return output


#for correlation 
def correlate(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(image, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    
    output = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i,j] = np.sum(region * kernel)
            
    return output

#template matching 
conv_result = convolve(image, template)

corr_result = correlate(image, template)

#best location 
conv_loc = np.unravel_index(np.argmax(conv_result), conv_result.shape)
corr_loc = np.unravel_index(np.argmax(corr_result), corr_result.shape)

print("Convolution location:", conv_loc)
print("Correlation location:", corr_loc)

#for rectangle around the identified location 
h, w = template.shape

image_color = cv2.imread("shelf.jpg")

# Convolution result
cv2.rectangle(image_color,
             (conv_loc[1]-w//2, conv_loc[0]-h//2),
             (conv_loc[1]+w//2, conv_loc[0]+h//2),
             (0,255,0), 2)

cv2.imwrite("conv_match.jpg", image_color)

# Correlation result
image_color2 = cv2.imread("shelf.jpg")

cv2.rectangle(image_color2,
             (corr_loc[1]-w//2, corr_loc[0]-h//2),
             (corr_loc[1]+w//2, corr_loc[0]+h//2),
             (255,0,0), 2)

cv2.imwrite("corr_match.jpg", image_color2)




# correlation is better because template matching requires exact pattern comparision 
# meanwhile convulation is used feature detection like edges ,blur etc so...
#Correlation is more suitable for template matching because it directly measures similarity 
#without flipping the template, making it more accurate and efficient than convolution.
#