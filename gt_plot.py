import numpy as np
import matplotlib.pyplot as plt
import json

anno_file = np.loadtxt("data/KETI/train_val.csv", dtype=int, delimiter=',')
# anno_file = np.loadtxt("PA100K_TEST_OUTPUT_NO_EMA.csv", dtype=float, delimiter=',')
#
# anno_file = (anno_file > 0.5).astype(np.int_)  # Where the numpy magic happens.

anno_file = np.sum(anno_file, axis=0)
# print(np.where(np.logical_and(np.greater(anno_file,1),np.less(anno_file,100)))[0])
# few = np.where(np.logical_and(np.greater(anno_file,1),np.less(anno_file,100)))[0]
few = np.where(anno_file>1)[0]
print(len(few))
names = np.arange(0, 81)
data = np.loadtxt("relation_key.csv", dtype=str, delimiter=',')
data = data[45:, 4].astype(int)
dictionary = {}
dictionary = dict(enumerate(data.flatten(), 0))
res = dict((v,k) for k,v in dictionary.items())

sparse = []
for i, item in enumerate(few):
    sparse.append(dictionary[item])
print(sparse)
plt.bar(names, anno_file)
plt.show()

fl = open('relation_key.json', 'r')
data = json.load(fl)

semantic = data["Attirbutes"]["semantic attributes"]["instance type"]
status = data["Attirbutes"]["semantic attributes"]["status"]
part = data["Attirbutes"]["semantic attributes"]["part"]
texture_material = data["Attirbutes"]["semantic attributes"]["texture-material"]
texture_status = data["Attirbutes"]["semantic attributes"]["texture-status"]
geometric = data["Attirbutes"]["geometric attributes"]

attributes = {}
attributes.update(semantic)
attributes.update(status)
attributes.update(part)
attributes.update(texture_material)
attributes.update(texture_status)
attributes.update(geometric)

reverse_attributes = dict((v,k) for k,v in attributes.items())

print(sparse)
name_few = []
for i, item in enumerate(sparse):
    print(item)
    name_few.append(reverse_attributes[item])

name_few.append('Lower-view')
name_few.append('drinking')
name_few.append('smoking')

attr_id = []
for i, item in enumerate(name_few):
    print(item)
    attr_id.append(attributes[item])
print(name_few)
print(len(name_few))
print(attr_id)
print(len(attr_id))

print(reverse_attributes)
print(attributes)

dictionary = dict(enumerate(attr_id, 0))
print(dictionary)