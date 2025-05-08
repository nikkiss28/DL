import numpy as np
image=np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 2, 2, 0],
    [2, 1, 0, 1, 2],
    [0, 1, 3, 1, 0]
])

kernel=np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

def convolution(image,kernel,stride=1):
    img_h,img_w=image.shape
    filt_h,filt_w=kernel.shape
    out_h=(image.shape[0]-filt_h)//stride+1
    out_w=(image.shape[1]-filt_w)//stride+1
    result=np.zeros((out_h,out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch=image[i*stride:i*stride+filt_h,j*stride:j*stride+filt_w]
            result[i,j]=np.sum(patch*kernel)
    return result

conv=convolution(image,kernel)
print(conv)

def max_pooling(image,pool_size=2,stride=1):
    img_h,img_w=image.shape
    out_h=(img_h-pool_size)//stride+1
    out_w=(img_w-pool_size)//stride+1
    result=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch=image[i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
            result[i,j]=np.max(patch)
    return result

pool=max_pooling(conv)
print(pool)

flat=pool.flatten()
print(flat)

def fully_connected(flat_inp,weight,bias):
    return np.dot(flat_inp,weights)+bias

weights=np.random.rand(flat.size,2)
bias=np.random.rand(2)

fc=fully_connected(flat,weights,bias)
print(fc)
