import numpy as np
def sigmoid(x):
  return 1/(1+np.exp(-x))

lr=0.01
latent=1
epochs=10000
w_g=np.random.rand()
b_g=2

w_d=np.random.rand()
b_d=0

for epoch in range(epochs):
  x_real=np.random.normal(2,0.5,(1,))
  z=np.random.rand(latent)
  x_fake=w_g*z+b_g

  d_real=sigmoid(w_d*x_real+b_d)
  d_fake=sigmoid(w_d*x_fake+b_d)
  d_loss=-np.log(d_real)-np.log(1-d_fake)

  grad_w_d=(d_real-1)*x_real+d_fake*x_fake
  grad_b_d=(d_real-1)+d_fake
  w_d-=lr*grad_w_d
  b_d-=lr*grad_b_d

  z=np.random.rand(latent)
  x_fake=w_g*z+b_g

  d_fake=sigmoid(w_d*x_fake+b_d)
  g_loss=-np.log(d_fake)

  grad_w_g=d_fake*(d_fake-1)*w_d*z
  grad_b_g=d_fake*(d_fake-1)*w_d

  w_g-=lr*grad_w_g
  b_g-=lr*grad_b_g

  if epoch%1000==0:
    print(f"epoch{epoch}, dloss={d_loss[0]}, gloss={g_loss[0]}")
print("Sample Gneneration: ")
for _ in range(5):
   z = np.random.randn(latent)
   sample = w_g * z + b_g
   print(sample)
