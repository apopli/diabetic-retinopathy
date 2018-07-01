from PIL import Image
import numpy as np
import os, sys

training_size=143
gt_dir = "gt/"
pred_dir = "predicted/"
sn = np.empty(training_size, dtype=float)
ppv = np.empty(training_size, dtype=float)
sp = np.empty(training_size, dtype=float)
image_paths = np.empty(training_size, dtype=object)

dir = os.getcwd()
i=0
for image_path in os.listdir(os.path.join(dir,gt_dir)):
	image_paths[i] = image_path
	im_gt = Image.open(gt_dir+image_path)
	im_pred = Image.open(pred_dir+image_path)
	arr_gt = np.array(im_gt)/255
	arr_pred = np.array(im_pred)/255
	
	true_p = np.sum(arr_gt & arr_pred)
	actual_p = np.sum(arr_gt)
	pred_p = np.sum(arr_pred)
	
	false_p = pred_p - true_p
	actual_n = 4288*2848 - actual_p
	true_n = actual_n - false_p
	# print "True pos: ", true_p
	# print "Actual pos: ", actual_p
	# print "Pred pos: ", pred_p
	if actual_p == 0:
		sn[i] = 1
	else:
		sn[i] = float(true_p)/float(actual_p)
	if pred_p == 0:
		ppv[i] = 1
	else:
		ppv[i] = float(true_p)/float(pred_p)
	print i
	if actual_n == 0:
		sp[i] = 1
	else:
		sp[i] = float(true_n)/float(actual_n)
	i+=1

sn_csv = np.stack((image_paths,sn), axis=1)
ppv_csv = np.stack((image_paths,ppv), axis=1)
sp_csv = np.stack((image_paths,sp), axis=1)

np.savetxt("sn.csv", sn_csv, delimiter=",", fmt="%s")
np.savetxt("ppv.csv", ppv_csv, delimiter=",", fmt="%s")
np.savetxt("sp.csv", sp_csv, delimiter=",", fmt="%s")