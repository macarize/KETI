import numpy as np

weights = []
for i in range(63):
    class_weight = np.loadtxt('weight_elem/obj_{}.csv'.format(i), dtype=float, delimiter=',')
    label_ratio = class_weight.mean(0)
    weights.append(label_ratio)

class_weight = np.stack(weights)
print(class_weight)
np.savetxt('class_weights.csv', class_weight, delimiter=',')