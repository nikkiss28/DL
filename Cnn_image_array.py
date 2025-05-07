image=np.array([
    [1,2,3,0,1],
    [0,1,2,3,1],
     [1, 0, 2, 2, 0],
    [2, 1, 0, 1, 2],
    [0, 1, 3, 1, 0]
])
filter1=np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])
def convolution(image,filter1,stride=1):
    img_h,img_w=image.shape
    filt_h,filt_w=filter1.shape
    out_h = (img_h - filt_h) // stride + 1
    out_w = (img_w - filt_w) // stride + 1
    result=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch=image[i*stride:i*stride+filt_h,j*stride:j*stride+filt_w]
            result[i,j]=np.sum(patch*filter1)
    return result
    def pooling(image,pool_size=2,stride=1):
    img_h,img_w=image.shape
    out_h=(img_h-pool_size)//stride+1
    out_w=(img_w-pool_size)//stride+1
    result1=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch=image[i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
            result1[i,j]=np.max(patch)
    return result1

conv=convolution(image,filter1)
print(conv)

pool=pooling(conv)
print(pool)

def fully_connected(flattend_inp,weights,bias):
    return np.dot(flattend_inp,weights)+bias

flattend=pool.flatten()
print(flattend)

weights = np.random.randn(flattend.size, 2)
bias = np.random.randn(2)

fc=fully_connected(flattend,weights,bias)
print(fc)
