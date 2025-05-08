import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
def circle(r,teta):
  x=r*np.cos(teta)
  y=r*np.sin(teta)
  return x,y

train=[]
for i in range(1000):
  teta=np.random.uniform(0,2*np.pi)
  r=5
  x,y=circle(r,teta)
  train.append([x,y])

X=np.array(train)

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

