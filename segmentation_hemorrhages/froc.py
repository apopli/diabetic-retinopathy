from PIL import Image
import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

training_size=143
gt_dir = "gt/"
prob_dir = "prob/"
true_p=0
actual_p=0
pred_p=0
false_p=0

thresh_list = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]

dir = os.getcwd()
thresh_size = len(thresh_list)
sn = np.empty(thresh_size, dtype=float)
fppi = np.empty(thresh_size, dtype=float)
thresh_array = np.array(thresh_list)

for th in range(thresh_size):
	threshold = thresh_array[th]
	print threshold
	true_p=0
	actual_p=0
	pred_p=0
	false_p=0

	for image_path in os.listdir(os.path.join(dir,gt_dir)):
		# print image_path
		im_gt = Image.open(gt_dir+image_path)
		im_prob = Image.open(prob_dir+image_path)
		arr_gt = np.array(im_gt)/255
		arr_prob = (np.array(im_prob)).astype(float)/255
		arr_pred = (arr_prob > threshold).astype('uint8')
		tp = np.sum(arr_gt & arr_pred)
		ap = np.sum(arr_gt)
		pp = np.sum(arr_pred)
		true_p += tp
		actual_p += ap
		pred_p += pp
		false_p += (pp-tp)

	sn[th] = float(true_p)/float(actual_p)
	print "sn: ", sn[th]
	fppi[th] = float(false_p)/float(training_size)
	print "fppi: ", fppi[th]

plt.plot(fppi, sn)
plt.ylabel('SN')
plt.xlabel('FPs per image')
plt.savefig('froc.png')