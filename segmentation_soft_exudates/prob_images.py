import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from PIL import Image
from scipy.special import expit
import os, sys

saver = tf.train.import_meta_graph("./models/model.ckpt.meta")
sess = tf.InteractiveSession()
saver.restore(sess, "models/model.ckpt")
X, mode = tf.get_collection("inputs")
pred = tf.get_collection("outputs")[0]

def read_image(image_path, gray=False):
    """Returns an image array

    Args:
        image_path (str): Path to image.jpg

    Returns:
        3-D array: RGB numpy image array
    """
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    image = cv2.imread(image_path)    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def pipeline(image, image_WH=(512, 512)):
    image = np.copy(image)
    H, W, C = image.shape
    
    if (W, H) != image_WH:
        image = cv2.resize(image, image_WH)
    
    mask_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, 0),
                                          mode: False})
    
    mask_pred = np.squeeze(mask_pred)
    mask_pred = expit(mask_pred)
    # mask_pred = mask_pred > threshold
    return mask_pred

output_dir = "prob2/"
dir = os.getcwd()

if not os.path.exists(os.path.join(dir,output_dir)):
    os.mkdir(output_dir)

for image_path in os.listdir(os.path.join(dir,"test_data2/")):
    image = read_image("test_data2/"+image_path)
    predicted_image = np.zeros((2848, 4288), dtype=float)

    for i in range(10):
        for j in range(16):
            top_y = i*256
            if (i==9):
                top_y = 2336
            top_x = j*256
            if (j==15):
                top_x = 3776

            image_crop = image[top_y:top_y+512, top_x:top_x+512]
            predicted_crop = pipeline(image_crop)
            predicted_image[top_y:top_y+512, top_x:top_x+512] = np.maximum(predicted_image[top_y:top_y+512, top_x:top_x+512], predicted_crop)

    # threshold = 0.5
    # predicted_image = predicted_image > threshold

    predicted_save = Image.fromarray((predicted_image*255).astype('uint8'))
    predicted_save.save(output_dir+image_path, "JPEG", quality=100) 