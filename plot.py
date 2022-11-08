import numpy as np
import matplotlib.pyplot as plt
sample = np.loadtxt("class_weights.csv", dtype=float, delimiter=",")
print(sample)

for i in range(63):
    ypoints = sample[i]

    plt.plot(ypoints, color = 'b')
    plt.savefig('obj_{}.png'.format(i))
    plt.show()