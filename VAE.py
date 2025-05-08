import tensorflow as tf
from tensorflow.keras import layers , models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train,y_train),(X_test, y_test)=mnist.load_data()
X_train=X_train/255
X_test=X_test/255
latent=2
inputs=layers.Input(shape=(28,28,1))

x=layers.Flatten()(inputs)
x=layers.Dense(128,activation='relu')(x)

z_mean=layers.Dense(latent)(x)
z_var=layers.Dense(latent)(x)

z=layers.Lambda(lambda args:args[0]+args[1]*tf.random.normal(tf.shape(args[1])))([z_mean,z_var])
decoder_h=layers.Dense(128,activation='relu')(z)
decoder_mean=layers.Dense(28*28,activation='sigmoid')(decoder_h)

vae_op=layers.Reshape((28,28,1))(decoder_mean)
vae=models.Model(inputs,vae_op)
vae.compile(loss='binary_crossentropy', optimizer='adam')
vae.fit(X_train,X_train,batch_size=128,epochs=10)
decoded=vae.predict(X_test)
plt.imshow(decoded[0])
plt.show()
