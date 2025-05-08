n = 250
x = np.linspace(0, 5, n)
y = np.linspace(0, 5, n)

rectangle = np.concatenate([
    np.stack([x, np.zeros(n)], axis=1),     
    np.stack([x, np.full(n, 5)], axis=1),   
    np.stack([np.zeros(n), y], axis=1),     
    np.stack([np.full(n, 5), y], axis=1)     
])
X=rectangle
X_scaled=StandardScaler().fit_transform(X)
pca=PCA(n_components=1)
X_pca=pca.fit_transform(X_scaled)
X_recons_pca=pca.inverse_transform(X_pca)
mse=mean_squared_error(X_scaled,X_recons_pca)
print(mse)

autoencoder=Sequential([
    Dense(1,activation='relu',input_shape=(2,)),
    Dense(2,activation='linear')
])

autoencoder.compile(loss='mse',optimizer='adam')
autoencoder.fit(X_scaled,X_scaled,epochs=100,batch_size=16,verbose=0)
encoder=Sequential(autoencoder.layers[:1])
X_encoded=encoder.predict(X_scaled)
X_ae_recons=autoencoder.predict(X_scaled)
mse_ae=mean_squared_error(X_scaled,X_ae_recons)
print(mse_ae)
