import numpy as np
W = np.array([[0.3, 0.6],
              [0.5, 0.7],
              [0.7, 0.4],
              [0.2, 0.3]])
inputs = np.array([
                    [1,0,1,0],
                    [1,0,0,0],
                    [1,1,1,1],
                    [0,1,1,0]
])
lr=0.6
r,c=W.shape
def euclidean(inp,W):
  distances=np.sum((W.T-inp)**2,axis=1)
  return distances.tolist()

epoch=0
while True:
  epoch+=1
  prev_W=W.copy()
  for inp in inputs:
    distances=euclidean(inp,W)
    win=np.argmin(distances)

    for i in range(r):
      W[i][win]+=lr*(inp[i]-W[i][win])
  W_rounded=np.round(W,4)
  prev_W_rounded=np.round(prev_W,4)
  print(f"\nEpoch {epoch} Weights:\n{W_rounded}")
  if np.array_equal(W_rounded, prev_W_rounded):
        break

print(f"\nConverged after {epoch} epochs.")
print("Final Weights:\n", W_rounded)
