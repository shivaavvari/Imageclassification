# -*- coding: utf-8 -*-
"""
This script demonstrates training a Convolutional Neural Network (CNN) to classify CIFAR10 images using TensorFlow and Keras. The script includes the following steps:

1. Import necessary libraries including TensorFlow, Keras, and Matplotlib.
2. Download and prepare the CIFAR10 dataset.
3. Verify the dataset by plotting sample images with their class names.
4. Create a convolutional base using Conv2D and MaxPooling2D layers.
5. Add Dense layers on top of the convolutional base for classification.
6. Compile and train the model.
7. Evaluate the model and plot the training history.
8. Create a new model using the VGG16 architecture pre-trained on ImageNet.
9. Freeze the layers of VGG16 and add custom Dense layers on top.
10. Compile and train the new model.
11. Evaluate the new model and plot the training history.

"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Import TensorFlow
import torch
import torchvision
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16



"""### Download and prepare the CIFAR10 dataset


The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each class. The dataset is divided into 50,000 training images and 10,000 testing images. The classes are mutually exclusive and there is no overlap between them.
"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

"""### Verify the data

To verify that the dataset looks correct, let's plot the first 25 images from the training set and display the class name below each image:

"""

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

"""### Create the convolutional base

The 6 lines of code below define the convolutional base using a common pattern: a stack of [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) and [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) layers.

As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. If you are new to these dimensions, color_channels refers to (R,G,B). In this example, you will configure your CNN to process inputs of shape (32, 32, 3), which is the format of CIFAR images. You can do this by passing the argument `input_shape` to your first layer.
"""

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

"""Let's display the architecture of your model so far:"""

model.summary()

"""Above, you can see that the output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels). The width and height dimensions tend to shrink as you go deeper in the network. The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64). Typically,  as the width and height shrink, you can afford (computationally) to add more output channels in each Conv2D layer.

### Add Dense layers on top

To complete the model, you will feed the last output tensor from the convolutional base (of shape (4, 4, 64)) into one or more Dense layers to perform classification. Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. First, you will flatten (or unroll) the 3D output to 1D,  then add one or more Dense layers on top. CIFAR has 10 output classes, so you use a final Dense layer with 10 outputs.
"""

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

"""Here's the complete architecture of your model:"""

model.summary()

"""The network summary shows that (4, 4, 64) outputs were flattened into vectors of shape (1024) before going through two Dense layers.

### Compile and train the model
"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

"""### Evaluate the model"""

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

# Predict on a single test image
single_test_image = test_images[1]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[1][0]]}')
plt.show()

# Get the prediction for the VGG16 model
vgg_predictions = new_model.predict(single_test_image)
vgg_predicted_class = class_names[tf.argmax(vgg_predictions[0])]

# Display the image and the predicted class for the VGG16 model
plt.figure(figsize=(1, 1)) 
plt.imshow(test_images[1])
plt.title(f'VGG16 Predicted: {vgg_predicted_class}')
plt.show()



# Create a new model using VGG16 with ImageNet weights
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of VGG16
for layer in vgg_model.layers:
  layer.trainable = False

# Add custom layers on top of VGG16
new_model = models.Sequential()
new_model.add(vgg_model)
new_model.add(layers.Flatten())
new_model.add(layers.Dense(64, activation='relu'))
new_model.add(layers.Dense(10))

# Compile the new model
new_model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

# Train the new model
new_history = new_model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))

# Evaluate the new model
plt.plot(new_history.history['accuracy'], label='accuracy')
plt.plot(new_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_test_loss, new_test_acc = new_model.evaluate(test_images, test_labels, verbose=2)

print(new_test_acc)


print(new_model.summary())

# Predict on a single test image
single_test_image = test_images[1]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[1][0]]}')
plt.show()

# Get the prediction for the VGG16 model
vgg_predictions = new_model.predict(single_test_image)
vgg_predicted_class = class_names[tf.argmax(vgg_predictions[0])]

# Display the image and the predicted class for the VGG16 model
plt.figure(figsize=(1, 1)) 
plt.imshow(test_images[1])
plt.title(f'VGG16 Predicted: {vgg_predicted_class}')
plt.show()








# Define the ResNet50 model with the new input shape
resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of ResNet50
for layer in resnet_model.layers:
  layer.trainable = False

# Add custom layers on top of ResNet50
new_resnet_model = models.Sequential()
new_resnet_model.add(resnet_model)
new_resnet_model.add(layers.Flatten())
new_resnet_model.add(layers.Dense(64, activation='relu'))
new_resnet_model.add(layers.Dense(10))

# Compile the new model
new_resnet_model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

# Train the new model
new_resnet_history = new_resnet_model.fit(train_images_resized, train_labels, epochs=10,
                      validation_data=(test_images_resized, test_labels))

# Evaluate the new model
plt.plot(new_resnet_history.history['accuracy'], label='accuracy')
plt.plot(new_resnet_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_resnet_test_loss, new_resnet_test_acc = new_resnet_model.evaluate(test_images, test_labels, verbose=2)

print(new_resnet_test_acc)


new_resnet_model.summary()
# Predict on a single test image
single_test_image = test_images[2]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[2][0]]}')
plt.show()
# Get the prediction for the ResNet50 model
resnet_predictions = new_resnet_model.predict(single_test_image)
resnet_predicted_class = class_names[tf.argmax(resnet_predictions[0])]

# Display the image and the predicted class for the ResNet50 model
plt.figure(figsize=(1, 1))

plt.imshow(test_images[2])
plt.title(f'ResNet50 Predicted: {resnet_predicted_class}')
plt.show()






# Create a new model using MobileNetV2 with ImageNet weights
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of MobileNetV2
for layer in mobilenet_model.layers:
  layer.trainable = False

# Add custom layers on top of MobileNetV2
new_mobilenet_model = models.Sequential()
new_mobilenet_model.add(mobilenet_model)
new_mobilenet_model.add(layers.Flatten())
new_mobilenet_model.add(layers.Dense(64, activation='relu'))
new_mobilenet_model.add(layers.Dense(10))

# Compile the new model
new_mobilenet_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the new model
new_mobilenet_history = new_mobilenet_model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

# Evaluate the new model
plt.plot(new_mobilenet_history.history['accuracy'], label='accuracy')
plt.plot(new_mobilenet_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_mobilenet_test_loss, new_mobilenet_test_acc = new_mobilenet_model.evaluate(test_images, test_labels, verbose=2)

print(new_mobilenet_test_acc)

# Predict on a single test image
single_test_image = test_images[3]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[3][0]]}')
plt.show()



# Get the prediction for the MobileNetV2 model
mobilenet_predictions = new_mobilenet_model.predict(single_test_image)
mobilenet_predicted_class = class_names[tf.argmax(mobilenet_predictions[0])]

# Display the image and the predicted class for the MobileNetV2 model
plt.imshow(test_images[3])
plt.title(f'MobileNetV2 Predicted: {mobilenet_predicted_class}')
plt.show()










# Create a new model using InceptionV3 with ImageNet weights
inception_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of InceptionV3
for layer in inception_model.layers:
  layer.trainable = False

# Add custom layers on top of InceptionV3
new_inception_model = models.Sequential()
new_inception_model.add(inception_model)
new_inception_model.add(layers.Flatten())
new_inception_model.add(layers.Dense(64, activation='relu'))
new_inception_model.add(layers.Dense(10))

# Compile the new model
new_inception_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the new model
new_inception_history = new_inception_model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

# Evaluate the new model
plt.plot(new_inception_history.history['accuracy'], label='accuracy')
plt.plot(new_inception_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_inception_test_loss, new_inception_test_acc = new_inception_model.evaluate(test_images, test_labels, verbose=2)

print(new_inception_test_acc)

# Predict on a single test image
single_test_image = test_images[4]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[4][0]]}')
plt.show()
# Get the prediction for the InceptionV3 model
inception_predictions = new_inception_model.predict(single_test_image)
inception_predicted_class = class_names[tf.argmax(inception_predictions[0])]

# Display the image and the predicted class for the InceptionV3 model
plt.imshow(test_images[4])
plt.title(f'InceptionV3 Predicted: {inception_predicted_class}')
plt.show()














# Create a new model using DenseNet121 with ImageNet weights
densenet_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of DenseNet121
for layer in densenet_model.layers:
  layer.trainable = False

# Add custom layers on top of DenseNet121
new_densenet_model = models.Sequential()
new_densenet_model.add(densenet_model)
new_densenet_model.add(layers.Flatten())
new_densenet_model.add(layers.Dense(64, activation='relu'))
new_densenet_model.add(layers.Dense(10))

# Compile the new model
new_densenet_model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

# Train the new model
new_densenet_history = new_densenet_model.fit(train_images, train_labels, epochs=10,
            validation_data=(test_images, test_labels))

# Evaluate the new model
plt.plot(new_densenet_history.history['accuracy'], label='accuracy')
plt.plot(new_densenet_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_densenet_test_loss, new_densenet_test_acc = new_densenet_model.evaluate(test_images, test_labels, verbose=2)

print(new_densenet_test_acc)


# Predict on a single test image
single_test_image = test_images[5]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[5][0]]}')
plt.show()

# Get the prediction for the DenseNet121 model
densenet_predictions = new_densenet_model.predict(single_test_image)
densenet_predicted_class = class_names[tf.argmax(densenet_predictions[0])]

# Display the image and the predicted class for the DenseNet121 model
plt.imshow(test_images[5])
plt.title(f'DenseNet121 Predicted: {densenet_predicted_class}')
plt.show()













# Create a new model using EfficientNetB0 with ImageNet weights
efficientnet_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of EfficientNetB0
for layer in efficientnet_model.layers:
  layer.trainable = False

# Add custom layers on top of EfficientNetB0
new_efficientnet_model = models.Sequential()
new_efficientnet_model.add(efficientnet_model)
new_efficientnet_model.add(layers.Flatten())
new_efficientnet_model.add(layers.Dense(64, activation='relu'))
new_efficientnet_model.add(layers.Dense(10))

# Compile the new model
new_efficientnet_model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

# Train the new model
new_efficientnet_history = new_efficientnet_model.fit(train_images, train_labels, epochs=10,
            validation_data=(test_images, test_labels))

# Evaluate the new model
plt.plot(new_efficientnet_history.history['accuracy'], label='accuracy')
plt.plot(new_efficientnet_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_efficientnet_test_loss, new_efficientnet_test_acc = new_efficientnet_model.evaluate(test_images, test_labels, verbose=2)

print(new_efficientnet_test_acc)

# Predict on a single test image
single_test_image = test_images[8]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[])
plt.title(f'Label: {class_names[test_labels[8][0]]}')
plt.show()




# Get the prediction for the EfficientNetB0 model
efficientnet_predictions = new_efficientnet_model.predict(single_test_image)
efficientnet_predicted_class = class_names[tf.argmax(efficientnet_predictions[0])]

# Display the image and the predicted class for the EfficientNetB0 model
plt.imshow(test_images[0])
plt.title(f'EfficientNetB0 Predicted: {efficientnet_predicted_class}')
plt.show()





# Create a new model using NASNetMobile with ImageNet weights
nasnet_model = tf.keras.applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of NASNetMobile
for layer in nasnet_model.layers:
  layer.trainable = False

# Add custom layers on top of NASNetMobile
new_nasnet_model = models.Sequential()
new_nasnet_model.add(nasnet_model)
new_nasnet_model.add(layers.Flatten())
new_nasnet_model.add(layers.Dense(64, activation='relu'))
new_nasnet_model.add(layers.Dense(10))

# Compile the new model
new_nasnet_model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

# Train the new model
new_nasnet_history = new_nasnet_model.fit(train_images, train_labels, epochs=10,
                      validation_data=(test_images, test_labels))

# Evaluate the new model
plt.plot(new_nasnet_history.history['accuracy'], label='accuracy')
plt.plot(new_nasnet_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_nasnet_test_loss, new_nasnet_test_acc = new_nasnet_model.evaluate(test_images, test_labels, verbose=2)

print(new_nasnet_test_acc)

# Predict on a single test image
single_test_image = test_images[7]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[7][0]]}')
plt.show()




# Get the prediction for the NASNetMobile model
nasnet_predictions = new_nasnet_model.predict(single_test_image)
nasnet_predicted_class = class_names[tf.argmax(nasnet_predictions[0])]

# Display the image and the predicted class for the NASNetMobile model
plt.imshow(test_images[7])
plt.title(f'NASNetMobile Predicted: {nasnet_predicted_class}')
plt.show()




# Create a new model using Xception with ImageNet weights
xception_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of Xception
for layer in xception_model.layers:
  layer.trainable = False

# Add custom layers on top of Xception
new_xception_model = models.Sequential()
new_xception_model.add(xception_model)
new_xception_model.add(layers.Flatten())
new_xception_model.add(layers.Dense(64, activation='relu'))
new_xception_model.add(layers.Dense(10))

# Compile the new model
new_xception_model.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

# Train the new model
new_xception_history = new_xception_model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

# Evaluate the new model
plt.plot(new_xception_history.history['accuracy'], label='accuracy')
plt.plot(new_xception_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

new_xception_test_loss, new_xception_test_acc = new_xception_model.evaluate(test_images, test_labels, verbose=2)

print(new_xception_test_acc)

# Predict on a single test image
single_test_image = test_images[0]
single_test_image = single_test_image.reshape(1, 32, 32, 3)  # Reshape to match the input shape

# Display a single test image
plt.figure(figsize=(1, 1))
plt.imshow(single_test_image[0])
plt.title(f'Label: {class_names[test_labels[0][0]]}')
plt.show()

# Get the prediction for the Xception model
xception_predictions = new_xception_model.predict(single_test_image)
xception_predicted_class = class_names[tf.argmax(xception_predictions[0])]

# Display the image and the predicted class for the Xception model
plt.figure(figsize=(1, 1))
plt.imshow(test_images[0])
plt.title(f'Xception Predicted: {xception_predicted_class}')
plt.show()


