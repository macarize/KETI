import json
import numpy as np
import os
import cv2
import matplotlib.image as mpimg

current_dir = os.getcwd()
files = os.listdir(current_dir)
json_files = [y for y in files if '.json' in y]
if not os.path.exists('out'):
    os.makedirs('out')

if not os.path.exists('saved'):
    os.makedirs('saved')
i = 0

if not os.path.exists('cropped'):
    os.makedirs('cropped')

###
import numpy as np
import matplotlib.pyplot as plt
import json

anno_file = np.loadtxt("train_val.csv", dtype=int, delimiter=',')
# anno_file = np.loadtxt("PA100K_TEST_OUTPUT_NO_EMA.csv", dtype=float, delimiter=',')
#
# anno_file = (anno_file > 0.5).astype(np.int_)  # Where the numpy magic happens.

anno_file = np.sum(anno_file, axis=0)
# print(anno_file)
# print(np.where(np.logical_and(np.greater(anno_file,1),np.less(anno_file,100)))[0])
# few = np.where(np.logical_and(np.greater(anno_file,1),np.less(anno_file,100)))[0]
few = np.where(anno_file>1)[0] #index of attr more than 1
# print(len(few))
names = np.arange(0, 149)
data = np.loadtxt("relation_key.csv", dtype=str, delimiter=',')
data = data[45:, 4].astype(int)
dictionary = {}
dictionary = dict(enumerate(data.flatten(), 0))
res = dict((v,k) for k,v in dictionary.items()) #original attribute id, index

sparse = []
for i, item in enumerate(few):
    sparse.append(dictionary[item])
# print(sparse)
plt.bar(names, anno_file)
plt.show()

fl = open('../meta_info/relation_key.json', 'r')
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

# print(sparse)
name_few = [] #subset attribute names
for i, item in enumerate(sparse):
    # print(item)
    name_few.append(reverse_attributes[item])

# name_few.append('Lower-view')
# name_few.append('drinking')
# name_few.append('smoking')

attr_id = [] #subset attribute id
for i, item in enumerate(name_few):
    # print(item)
    attr_id.append(attributes[item])
# print(name_few)
# print(len(name_few))
# print(attr_id)
# print(len(attr_id))



###



def order_points(pts):
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped
count = 0
labeled_person_count = 0
female_male = 0
idx = 0

import numpy as np

data = np.loadtxt("relation_key.csv", dtype=str, delimiter=',')
data = data[45:, 4].astype(int)
dictionary = {}
dictionary = dict(enumerate(data.flatten(), 0))
# print(dictionary)
# dictionary = dict(enumerate(attr_id, 0)) #index, subset attribute id
# print(dictionary)


res = dict((v,k) for k,v in dictionary.items())
classes = []
for i in range(63):
	classes.append([])

labels = []
classes_ids = []
for f in json_files: #image
	print('trying to read:',f)
	fl = open(f,'r')
	data = json.load(fl)
	image_name = f[:-4]+'png'
	if not os.path.exists(image_name):
		image_name = f[:-4]+'jpg'
	image = cv2.imread('../image/' + image_name)
	boxes = data["objects"]
	image_width, image_height = image.shape[0], image.shape[1]
	for i,item in enumerate(boxes): #per person
		# if item["class_name"] == 'person':
			# label = np.zeros(149)
			label = np.zeros(149)

			duplicate = 0
			pts = item["object_bbox"]
			x_min, y_min, x_max, y_max = pts['x'], pts['y'], pts['x'] + pts['width'], pts['y'] + pts['height']

			cropped_img = image[max(0, y_min): max(0, y_max), max(0, x_min): max(0, x_max)].copy()
			if cropped_img.size != 0 :
				cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
				# mpimg.imsave(os.path.join('cropped',f[:-4]+'_'+str(i)+'.jpg'),cropped_img)
				# mpimg.imsave(os.path.join('all_obj',str(idx)+'_'+str(i)+'.jpg'),cropped_img)
				file_name = str(idx)+'_'+str(i)+'.jpg'
				count += 1
				# print(i+1) #person id
				for k in range(len(data["predicates"])):
					if data["predicates"][k]["subject_id"] == i+1:
						if data["predicates"][k]["object_id"] == -1:
							# print(data["predicates"][k])
							id = data["predicates"][k]["predicate_id"]
							if int(id) in res:
								vector_index = res[int(id)]
								label[vector_index] = 1
				# print(label)
				labels.append(label)
				obj_id = item["class_id"]
				obj_index = obj_id - 1
				classes[int(obj_index)].append(label.tolist())
				classes_ids.append(obj_index)
	idx += 1
	#cv2.polylines(image,[pts],True,(0,0,255))
	#cv2.imshow('window',image)
	#cv2.waitKey(5000)
	# if i <= 5:
	# 	cv2.imwrite('saved/'+image_name,image)
	# i+=1
labels = np.stack(labels, axis=0)
# np.savetxt('gt_01.csv', labels, delimiter=',')
print(labeled_person_count)
print(count)
print(female_male)
np.savetxt('classes_ids.csv', classes_ids, delimiter=',')
for i in range(len(classes)):
	print(i)
	obj = classes[i]

	obj = np.array(obj)
	print(obj)
	np.savetxt('obj_{}.csv'.format(i), obj, delimiter=',')

# with open('KETI_01_index_attr.json', 'w') as fp:
# 	json.dump(dictionary, fp)