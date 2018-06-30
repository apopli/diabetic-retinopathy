from PIL import Image
import numpy as np
import os, sys
from sklearn.metrics import average_precision_score

training_size=143
gt_dir = "gt/"
prob_dir = "prob/"

dir = os.getcwd()

i=0
sum_pav=0
for image_path in os.listdir(os.path.join(dir,gt_dir)):
	# print image_path
	im_gt = Image.open(gt_dir+image_path)
	im_prob = Image.open(prob_dir+image_path)
	arr_gt = (np.array(im_gt)/255).astype(bool)
	arr_prob = (np.array(im_prob)).astype(float)/255
	pav = average_precision_score(arr_gt.reshape((-1)),arr_prob.reshape((-1)))
	sum_pav = sum_pav+pav
	i = i+1

mpav = sum_pav/i
print mpav