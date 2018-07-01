# Segmentation of Hard Exudates
This directory comprises code to preprocess, train, test and evaluate our model. The implementation is based on Tensorflow.

## Requirements
You need to have the following python packages installed (may be incomplete):

* tensorflow
* opencv
* numpy
* scipy
* sklearn
* matplotlib
* PIL
* pandas

You need to have downloaded at least the IDRiD training dataset.

## How to use

### Set paths
We need to know where you saved the data and where you wish to save the results. Go into the respective python scripts and modify the paths 
to suit your setup.

### Data preprocessing
The ground truths are thresholded to levels 0 and 255.

```
python gt_binary.py
```

Since the retinal image is of a very high resolution, smaller-sized overlapping patches are extracted from the training images and masks. This
is done while maintaining the ratio of patches with lesions do those without lesions.

```
python extract_patches.py
```

The patches are separated into training and validation data, and the respective file names are written in train.csv and test.csv.

```
python traintest_split.py
```

### Network training
Now train the UNet model on the dataset of patches. In order to also validate on test split while training

```
python train.py --epochs=100
```

To train without validation

```
python train_100.py --epochs=100
```

### Network testing
To test the trained model on a single patch

```
python test.py
```

To test on an entire image

```
python test_image.py
```

Generate probability maps for the dataset of retinal images using the trained model

```
python prob_images.py
```

Generate segmented output masks from the probability maps

```
python test_images.py
```

### Evaluation

Calculate sensitivity and precison values for individual images, plot FROC curve, compute average statistics and precision score

```
python sn_ppv.py
python froc.py
python stats_avg.py
python score.py
```
