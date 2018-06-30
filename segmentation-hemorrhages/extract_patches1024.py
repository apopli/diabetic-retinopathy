from PIL import Image
from resizeimage import resizeimage
import os, sys
import numpy as np

dir = os.getcwd()
output_dir_data = "patches/"
output_dir_mask = "labels/"
if not os.path.exists(os.path.join(dir,output_dir_data)):
    os.mkdir(output_dir_data)
if not os.path.exists(os.path.join(dir,output_dir_mask)):
    os.mkdir(output_dir_mask)

dir_data = os.path.join(dir,"data/")
dir_mask = os.path.join(dir,"mask/")

# im = Image.open(os.path.join(dir_mask,"IDRiD_06.tif"))
# im_crop = im.crop((2000,0,2000+512,0+256))
# im_crop.show()
# image_np = np.array(im_crop)
# print np.sum(image_np)

negative_patches = []
positive_count = 0

for file in os.listdir(dir_mask):
    outfile = os.path.splitext(file)[0]
    extension = os.path.splitext(file)[1]
    if (cmp(extension, ".jpg")):
        continue

    im = Image.open(os.path.join(dir_mask,file))
    imd = Image.open(os.path.join(dir_data,file))
    # image_np = np.array(im)
    # print np.sum([True, True])
    # im_crop = im.crop((1900,0,1900+512,0+512))
    patch_id = 0
    for i in range(5):
    	for j in range(8):
            top_y = i*512
            if (i==4):
                top_y = 1824
            top_x = j*512
            if (j==7):
                top_x = 3264

            im_crop = im.crop((top_x,top_y,top_x+1024,top_y+1024))
            imd_crop = imd.crop((top_x,top_y,top_x+1024,top_y+1024))
            if (np.sum(np.array(im_crop)) < 100):
                negative_patches.append(output_dir_mask+outfile+"_p"+str(patch_id)+extension)
            else:
                positive_count += 1
            # im_crop = resizeimage.resize_cover(im_crop, [512, 512])
            # imd_crop = resizeimage.resize_cover(imd_crop, [512, 512])
            im_crop.save(output_dir_mask+outfile+"_p"+str(patch_id)+extension,"JPEG",quality=100)
            imd_crop.save(output_dir_data+outfile+"_p"+str(patch_id)+extension,"JPEG",quality=100)

            patch_id += 1

negative_patches = np.array(negative_patches)
# np.savetxt("negative.csv", negative_patches, delimiter=",", fmt="%s")

negative_count = negative_patches.size
delete_count = negative_count - positive_count
np.random.shuffle(negative_patches)
split_idx = delete_count
delete_patches = negative_patches[:split_idx]

for idx in range(delete_patches.size):
    os.remove(delete_patches[idx])
    os.remove("patches"+delete_patches[idx][6:])