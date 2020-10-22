# Object detection Dataset Generator

## Features 
This repo's goal is to generate simple datasets to test object detection models.
In particular it allows to:
* Decide the image size
* Decide how many object to include in each image
  * Fixed number or maximum number, and randomly extract the number of objects for each image
  * Decide the type of objects to include in the image:
    * Classification datasets from `torchvision.dataset` like MNIST, FashionMNIST etc..
  * Decide the size of the objects to include in the image
  * export ground truths bbox in yolo unnormalized format

### TODO  
* Allow the use of different types of background
* Allow simple iamge transformations for included objects (eg: rotations..)
* Allow other export formats for annotations


## Samples

### MNIST
![sample](assets/raw_1obj.png)
![sample2](assets/raw_4obj.png)

and with the their ground truth

![sample3](assets/gg_3obj.png)

## Command line interface

Command line parameters  
* -s  image size (int) (squared image)
* -o  object size (int) max size of enlcosing square for objects inserted in the image
* -m  maximum numbe of objects in each iamge
* -n  total number of images to be generated (it will be split in train and validation 80/20)
* -f  annotation format, if yolo the the coordinates will be normalized with iamge size value
