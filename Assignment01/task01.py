import cv2
import numpy as np


image = cv2.imread('dog.jpg')  

image_float = image.astype(np.float32)

mean = 0
std_dev = 15
gaussian_noise = np.random.normal(mean, std_dev, image.shape)

noisy_image = image_float + gaussian_noise

noisy_image = np.clip(noisy_image, 0, 255)


noisy_image = noisy_image.astype(np.uint8)
cv2.imwrite('noisy_image.jpg', noisy_image)
print("Noisy image saved as noisy_image.jpg")