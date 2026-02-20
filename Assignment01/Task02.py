import cv2
import numpy as np

image = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

pad_h = kernel.shape[0] // 2
pad_w = kernel.shape[1] // 2

padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

flipped_kernel = np.flip(kernel)

output = np.zeros_like(image, dtype=np.float32)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        
        region = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
        
        output[i, j] = np.sum(region * flipped_kernel)


output = np.clip(output, 0, 255)
output = output.astype(np.uint8)


cv2.imwrite("convolved_image.jpg", output)

print("Convolution completed and image iss saved!")