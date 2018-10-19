Multi-task classification example in Gluon
=============================================

This tutorial shows how to do a multi-task classification using Gluon.
Here we have used a curated fashion dataset containing apparels of different colour and we want to 
build a classifier to predict the category and color of th apparel. 
The network uses two loss functions and also uses two outputs and hence serves as a good example
of multi-task classification.

Dataset Attribution:
You can download the dataset from Fashion-dataset. The dataset has been curated and developed as a part of 
this blogpost and comes under MIT opensource license. The dataset contains 7 classes of images:
```r
1. Black jeans
2. Black shoes
3. Blue dress
4. Blue Jeans
5. Blue shirts
6. Red dress
7. Red shirts
```
The dataset has been divided into train and val folder. Train folder contains 300 images for each of the classes
whereas val folder contains ~70-75 images for each of the clasees making train-val split to be roughly 80-20.

Loading required packages
---------
Lets start with loading required packages

```r
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)
from mxnet.gluon.data.vision import transforms
import os
```

Prepare  the labels
---------
```r

'''
Prepare labels from the directory structure of the dataeset. The folder for each of the classes is named as
'category_color'. We will split the label and  the color and category labels to an index required for inference.
We also map the combined label index for category and colors to original labels which will be required for accuracy 
calculation
'''
##path of training dataset where you downloaded the dataset,
path = '/Users/khedia/Downloads/fashionnet_gluon/dataset/train'
labels = os.listdir(path)
categoryLabels = []
colorLabels = []
color_label_dict= {}
category_label_dict={}
combined_label_dict={}
i=0
j=0
k=0
for path in labels:
    (color, cat) = path.split("_")
    if not cat in category_label_dict.keys():
        category_label_dict[cat] = i
        i+=1
    categoryLabels.append(category_label_dict[cat])
    if not color in color_label_dict.keys():
        color_label_dict[color]=j
        j+=1
    colorLabels.append(color_label_dict[color])
    combined_label_dict[str(category_label_dict[cat])+str(color_label_dict[color])]= k
    k+=1
```



Pre-process datsets and prepare dataloader
---------

```r
batch_size = 32
num_gpus = 0
num_workers = 4
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
# path to the dataset
path = '/Users/khedia/Downloads/fashionnet_gluon/dataset'
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')

#resize images to 3X96X96, we have used transform functions from gluoncv for this pre-processing step 
transform_train = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
])

## Create dataloader for training as well as validation set
train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

```
Now, we create two parallel convolutional networks for classifying category and colors
```r
## build category branch
num_category = len(set(categoryLabels))
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.MaxPool2D(pool_size=3))
    net.add(gluon.nn.Dropout(0.25))
    
    net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.MaxPool2D(pool_size=2))
    net.add(gluon.nn.Dropout(0.25))
    
    net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, activation='relu'))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, activation='relu'))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.MaxPool2D(pool_size=2))
    net.add(gluon.nn.Dropout(0.25))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Dropout(0.5))
    net.add(gluon.nn.Dense(num_category))
    

```

```r 
num_colors = len(set(colorLabels))
net1 = gluon.nn.Sequential()
with net1.name_scope():
    net1.add(gluon.nn.Conv2D(channels=16, kernel_size=3, activation='relu'))
    net1.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net1.add(gluon.nn.MaxPool2D(pool_size=3))
    net1.add(gluon.nn.Dropout(0.25))
    
    net1.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
    net1.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net1.add(gluon.nn.MaxPool2D(pool_size=2))
    net1.add(gluon.nn.Dropout(0.25))
    
    net1.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
    net1.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net1.add(gluon.nn.MaxPool2D(pool_size=2))
    net1.add(gluon.nn.Dropout(0.25))
    

    net1.add(gluon.nn.Flatten())
    net1.add(gluon.nn.Dense(128, activation="relu"))
    net1.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net1.add(gluon.nn.Dropout(0.5))
    net1.add(gluon.nn.Dense(num_colors))
```

Please note that the network for color classification is quite shallow compared to the category one as learning category is tougher task than learning color.
Next we initilaize parameters and tariner for the networks.
```r
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
net1.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
trainer1 = gluon.Trainer(net1.collect_params(), 'sgd', {'learning_rate': .1})

```


