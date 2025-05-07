import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def forward_pass(inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4):
    net_h1 = inputs[0] * w1 + inputs[1] * w2 + b1
    out_h1 = sigmoid(net_h1)

    net_h2 = inputs[0] * w3 + inputs[1] * w4 + b2
    out_h2 = sigmoid(net_h2)

    net_o1 = out_h1 * w5 + out_h2 * w6 + b3
    out_o1 = sigmoid(net_o1)

    net_o2 = out_h1 * w7 + out_h2 * w8 + b4
    out_o2 = sigmoid(net_o2)

    E_total = 0.5 * ((targets[0] - out_o1) ** 2 + (targets[1] - out_o2) ** 2)

    return net_h1, out_h1, net_h2, out_h2, net_o1, out_o1, net_o2, out_o2, E_total

def update_output_weights(out_h1, out_h2, out_o1, out_o2, targets, w5, w6, w7, w8, learning_rate):
  #we want neww5 and so on
    dE_o1 = (out_o1-targets[0])
    dE_o2 = (out_o2-targets[1])

    d_out_o1_net_o1 = out_o1 * (1 - out_o1)
    d_out_o2_net_o2 = out_o2 * (1 - out_o2)

    #d_net_w5=out_h1
    dE_w5 = dE_o1 * d_out_o1_net_o1 * out_h1
    dE_w6 = dE_o1 * d_out_o1_net_o1 * out_h2
    dE_w7 = dE_o2 * d_out_o2_net_o2 * out_h1
    dE_w8 = dE_o2 * d_out_o2_net_o2 * out_h2

    new_w5 = w5 - learning_rate * dE_w5
    new_w6 = w6 - learning_rate * dE_w6
    new_w7 = w7 - learning_rate * dE_w7
    new_w8 = w8 - learning_rate * dE_w8

    return new_w5, new_w6, new_w7, new_w8, dE_o1, dE_o2, d_out_o1_net_o1, d_out_o2_net_o2

def update_hidden_weights(inputs, out_h1, out_h2, dE_o1, dE_o2, d_out_o1_net_o1, d_out_o2_net_o2, w5, w6, w7, w8, learning_rate):
    d_out_h1_net_h1 = out_h1 * (1 - out_h1)
    d_out_h2_net_h2 = out_h2 * (1 - out_h2)

    dE_o1_d_w1 = dE_o1 * d_out_o1_net_o1 * w5 * d_out_h1_net_h1 * inputs[0]
    dE_o2_d_w1 = dE_o2 * d_out_o2_net_o2 * w7 * d_out_h1_net_h1 * inputs[0]
    dE_total = dE_o1_d_w1 + dE_o2_d_w1

    dE_o1_d_w2 = dE_o1 * d_out_o1_net_o1 * w6 * d_out_h2_net_h2 * inputs[1]
    dE_o2_d_w2 = dE_o2 * d_out_o2_net_o2 * w8 * d_out_h2_net_h2 * inputs[1]
    dE_total2 = dE_o1_d_w2 + dE_o2_d_w2

    dE_o1_d_w3 = dE_o1 * d_out_o1_net_o1 * w5 * d_out_h1_net_h1 * inputs[0]
    dE_o2_d_w3 = dE_o2 * d_out_o2_net_o2 * w7 * d_out_h1_net_h1 * inputs[0]
    dE_total3 = dE_o1_d_w3 + dE_o2_d_w3

    dE_o1_d_w4 = dE_o1 * d_out_o1_net_o1 * w6 * d_out_h2_net_h2 * inputs[1]
    dE_o2_d_w4 = dE_o2 * d_out_o2_net_o2 * w8 * d_out_h2_net_h2 * inputs[1]
    dE_total4 = dE_o1_d_w4 + dE_o2_d_w4

    w1_new = w1 - learning_rate * dE_total
    w2_new = w2 - learning_rate * dE_total2
    w3_new = w3 - learning_rate * dE_total3
    w4_new = w4 - learning_rate * dE_total4

    return w1_new, w2_new, w3_new, w4_new

# Input and weight initialization
inputs = np.array([0.1, 0.5])
targets = np.array([0.05, 0.95])

w1, w2 = 0.1, 0.3
w3, w4 = 0.2, 0.4
w5, w6 = 0.5, 0.6
w7, w8 = 0.7, 0.8

b1, b2 = 0.25, 0.25
b3, b4 = 0.35, 0.35

learning_rate = 0.6

# Forward pass
net_h1, out_h1, net_h2, out_h2, net_o1, out_o1, net_o2, out_o2, E_total = forward_pass(
    inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4)

print("Forward Pass:")
print(f"Net h1: {net_h1}, Output h1: {out_h1}")
print(f"Net h2: {net_h2}, Output h2: {out_h2}")
print(f"Net o1: {net_o1}, Output o1: {out_o1}")
print(f"Net o2: {net_o2}, Output o2: {out_o2}")
print(f"Total Error: {E_total}")

# Backpropagation: Output layer
new_w5, new_w6, new_w7, new_w8, dE_o1, dE_o2, d_out_o1_net_o1, d_out_o2_net_o2 = update_output_weights(
    out_h1, out_h2, out_o1, out_o2, targets, w5, w6, w7, w8, learning_rate)

print("\nBackward Pass:")
print(f"Updated Weights:  w5={new_w5}, w6={new_w6}, w7={new_w7}, w8={new_w8}")

# Backpropagation: Hidden layer
w1_new, w2_new, w3_new, w4_new = update_hidden_weights(
    inputs, out_h1, out_h2, dE_o1, dE_o2, d_out_o1_net_o1, d_out_o2_net_o2, w5, w6, w7, w8, learning_rate)

print("Updated Weights:")
print(f"w1={w1_new}, w2={w2_new}, w3={w3_new}, w4={w4_new}")
