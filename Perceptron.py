#AND USING BIPOLAR
def fnet(net):
  if net>0:
    return 1
  elif net<0:
    return -1
  else:
    return 0

w1=0
w2=0
b=0
c=1
x1=[-1,-1,1,1]
x2=[-1,1,-1,1]
d=[-1,-1,-1,1]
epochs=10
for epoch in range(epochs):
  print(f"epoch{epoch+1}")
  error_count=0
  for i in range(4):
    net=w1*x1[i]+w2*x2[i]+b
    O=fnet(net)
    e=d[i]-O
    delw1=c*e*x1[i]
    delw2=c*e*x2[i]
    delb=e
    w1+=delw1
    w2+=delw2
    b+=delb
    print(f"x1 = {x1[i]:2} | x2 = {x2[i]:2} | d = {d[i]:2} | net = {net:3} | y = {O:2} | e = {e:2} | " f"Δw1 = {delw1:2} | Δw2 = {delw2:2} | Δb = {delb:2} | w1 = {w1:2} | w2 = {w2:2} | b = {b:2}")
    if e!=0:
      error_count+=1

  if error_count==0:
    print("Training converged")
    break
