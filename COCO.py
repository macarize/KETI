from dataset.multi_label.coco import download_coco2014
import os
import pickle
# download_coco2014('D:\Rethinking_of_PAR-master (2)\Rethinking_of_PAR-master\COCO','train')

list_path = os.path.join('D:\Rethinking_of_PAR-master (2)\Rethinking_of_PAR-master\data\COCO14', 'ml_anno', f'coco14_train_anno.pkl')
anno = pickle.load(open(list_path, 'rb+'))
print(len(anno['labels']))

import csv

with open("COCO_train_anno.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(anno['labels'])