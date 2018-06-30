from PIL import Image
import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gt_dir = "gt/"
pred_dir = "predicted/"
true_p=0
actual_p=0
pred_p=0
false_p=0
actual_n=0
true_n=0

dir = os.getcwd()

for image_path in os.listdir(os.path.join(dir,gt_dir)):
	print image_path
	im_gt = Image.open(gt_dir+image_path)
	im_pred = Image.open(pred_dir+image_path)
	arr_gt = np.array(im_gt)/255
	arr_pred = np.array(im_pred)/255

	tp = np.sum(arr_gt & arr_pred)
	ap = np.sum(arr_gt)
	pp = np.sum(arr_pred)

	fp = pp - tp
	an = 4288*2848 - ap
	tn = an - fp

	true_p += tp
	actual_p += ap
	pred_p += pp
	false_p += fp
	actual_n += an
	true_n += tn


sn = float(true_p)/float(actual_p)
ppv = float(true_p)/float(pred_p)
sp = float(true_n)/float(actual_n)

print "SN: ", sn
print "PPV: ", ppv
print "SP: ", sp