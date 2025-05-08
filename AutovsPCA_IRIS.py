from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=load_iris()
X=data.data
X_scaled=StandardScaler().fit_transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
X_recons_pca=pca.inverse_transform(X_pca)
mse=mean_squared_error(X_scaled,X_recons_pca)
print(mse)

plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:,0], X_pca[:,1],cmap='viridis',c=data.target)
plt.title("PCA (2D) Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

autoencoder=Sequential([
    Dense(2,activation='relu',input_shape=(4,)),
    Dense(4,activation='linear')
])

autoencoder.compile(loss='mse',optimizer='adam')
autoencoder.fit(X_scaled,X_scaled,epochs=100,batch_size=16,verbose=0)
encoder=Sequential(autoencoder.layers[:1])
X_encoded=encoder.predict(X_scaled)
X_ae_recons=autoencoder.predict(X_scaled)
mse_ae=mean_squared_error(X_scaled,X_ae_recons)
print(mse_ae)

plt.figure(figsize=(6, 4))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=data.target, cmap='viridis')
plt.title("PCA (2D) Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
