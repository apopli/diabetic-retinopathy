import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from PIL import Image
from scipy.special import expit

saver = tf.train.import_meta_graph("./models/model.ckpt.meta")
sess = tf.InteractiveSession()
saver.restore(sess, "models/model.ckpt")
X, mode = tf.get_collection("inputs")
pred = tf.get_collection("outputs")[0]

np.set_printoptions(threshold=np.nan)

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

def pipeline(image, threshold=0.5, image_WH=(512, 512)):
    image = np.copy(image)
    H, W, C = image.shape
    
    if (W, H) != image_WH:
        image = cv2.resize(image, image_WH)
    
    mask_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, 0),
                                          mode: False})
    
    mask_pred = np.squeeze(mask_pred)
    mask_pred = expit(mask_pred)
    mask_pred = mask_pred > threshold
    
    # labeled_heatmap, n_labels = label(mask_pred)
    
    # bbox = []
    
    # for i in range(n_labels):
    #     mask_i = labeled_heatmap == (i + 1)
        
    #     nonzero = np.nonzero(mask_i)
        
    #     nonzero_row = nonzero[0]
    #     nonzero_col = nonzero[1]
        
    #     left_top = min(nonzero_col), min(nonzero_row)
    #     right_bot = max(nonzero_col), max(nonzero_row)
        
    #     if not already_drawn_bbox(bbox, left_top, right_bot):
    #         image = cv2.rectangle(image, left_top, right_bot, color=(0, 255, 0), thickness=3)
        
    #         bbox.append((left_top, right_bot))
    
    # return image
    return mask_pred

image_path = "data_resize/IDRiD_14_p121.jpg"
image = read_image(image_path)

predicted = pipeline(image)
img = Image.fromarray((predicted.astype('uint8'))*255)
# imrgb = Image.merge('RGB', (img,img,img))

# image.save("Mask.jpg", "JPEG")
img.save("Predicted.jpg", "JPEG")

# print (predicted.astype(int))*255