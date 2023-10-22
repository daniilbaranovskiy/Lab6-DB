import numpy as np
import neurolab as nl

target = [[0, 1, 1, 1, 0,
           0, 0, 1, 0, 0,
           0, 1, 0, 1, 0,  # B
           0, 1, 1, 1, 0,
           0, 1, 1, 1, 0],

          [0, 0, 0, 0, 0,
           0, 1, 0, 1, 0,
           1, 1, 0, 1, 1,  # D
           0, 1, 0, 1, 0,
           1, 0, 1, 0, 1],

          [0, 1, 0, 1, 0,
           1, 1, 0, 0, 1,
           1, 0, 1, 0, 1,  # S
           1, 0, 0, 1, 1,
           0, 1, 0, 1, 0]

          ]
chars = ['B', 'D', 'S']
target = np.asfarray(target)
target[target == 0] = -1
net = nl.net.newhop(target)
output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())
print("\nTest on defaced S:")
test = np.asfarray(
    [0, 1, 1, 1, 1,
     1, 1, 1, 0, 1,
     1, 0, 1, 0, 1,  # S
     1, 0, 1, 1, 1,
     0, 1, 1, 1, 1],
)
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))


