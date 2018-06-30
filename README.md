# Overview
This repository provides source code, submitted papers and demo for Diabetic Retinopathy: Segmentation, Grading and Localization
with IDRiD dataset.Our method won the 1st place in Fovea Localization with overall 3rd place in the Localization sub-challenge of
IDRiD Grand Challenge. Besides, we secured 5th rank in Segmentation of Hard Exudates.

# Lesion Segmentation
We developed a data-driven method for automatic detection of retinal lesions based on their characteristics in fundus images to treat
Diabetic Retinopathy. First we preprocessed retinal images to reduce image noise and did several data augmentations to make data variety more
richer and distinct. After that, we segmented them using a model with UNet architecture and weighted cross entropy loss. The Unet architecture
includes a shrinking path to capture context of surrounding and a symmetric expanding path that enables accurate localization. We improved
and extended this architecture such that it works with very few training images and produces more accurate lesion masks. We have trained four
distinct binary models for the four lesions - hard exudates, hemorrhages, microaneurysms and soft exudates.

# Disease Grading
The retina images for disease classification were obtained from Kaggle dataset. The labels were provided by clinicians who rated
the severity of diabetic retinopathy in each image on a scale of 0-4. The images were preprocessed, downsampled and augmented before feeding
them to a 50 layer deep ResNet. Our networks achieved very competitive Kappa score of 0.74 on Kaggle Private Leaderboard, along with
sensitivity of 82% and specificity of 84%.

# Localization
The model consisted of two subsequent approaches to do localization. The initial model was a Convolution based model to get a tentative
position of the Fovia and the Optical Center. The next model worked on a patch around the predicted value of the previous model to improve 
the accuracy. The motivation for sucha step was to first get a global view of the image and then focus on a local area for predicting
the coordinates. The first model is a standard CNN while the second model implements UNet architecture.

-------------------------------------------------------------------------

The repository also provides <b>short papers</b> submitted to IEEE ISBI as part of the IDRiD Grand Challenge.
