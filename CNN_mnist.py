import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (_, _) = mnist.load_data()

image = X_train[0] / 255.0
label = y_train[0]
print(f"True Label: {label}")

plt.imshow(image, cmap='gray')
plt.title(f"Digit: {label}")
plt.show()

filter1=np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

# Step 3: Convolution
def convolution(image, filter1, stride=1):
    img_h, img_w = image.shape
    filt_h, filt_w = filter1.shape
    out_h = (img_h - filt_h) // stride + 1
    out_w = (img_w - filt_w) // stride + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+filt_h, j*stride:j*stride+filt_w]
            result[i, j] = np.sum(patch * filter1)
    return result

# Step 4: Max Pooling
def pooling(image, pool_size=2, stride=2):
    img_h, img_w = image.shape
    out_h = (img_h - pool_size) // stride + 1
    out_w = (img_w - pool_size) // stride + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            result[i, j] = np.max(patch)
    return result

# Step 5: Fully Connected Layer
def fully_connected(flattened_input, weights, bias):
    return np.dot(flattened_input, weights) + bias

# Pipeline
conv = convolution(image, filter1)
pool = pooling(conv)
flattened = pool.flatten()

# Create weights and bias for 10 output classes
weights = np.random.randn(flattened.shape[0], 10)
bias = np.random.randn(10)

fc_output = fully_connected(flattened, weights, bias)
predicted = np.argmax(fc_output)

print("Predicted class index:", predicted)
