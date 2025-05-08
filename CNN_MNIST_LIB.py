import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (_, _) = mnist.load_data()

image = X_train[0] / 255.0
label = y_train[0]
print(f"True Label: {label}")

plt.imshow(image, cmap='gray')
plt.title(f"Digit: {label}")
plt.show()

patch = image[:5, :5].reshape(5, 5, 1)

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

model = Sequential([
    Conv2D(filters=1, kernel_size=(3, 3), padding='valid', activation='linear',
           input_shape=(5, 5, 1), use_bias=False),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=1, activation='sigmoid')
])

model.layers[0].set_weights([kernel.reshape(3, 3, 1, 1)])
model.compile(optimizer='adam', loss='mean_squared_error')

output = model.predict(patch.reshape(1, 5, 5, 1))
print(f"The output value is {output}")
