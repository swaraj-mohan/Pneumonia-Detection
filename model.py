#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from glob import glob

#resize all images to the expected size
image_size = [224, 224]

#set train, test, validation dataset path
train_path = '/content/drive/My Drive/Datasets/Pneumonia Chest Xray/train'
valid_path = '/content/drive/My Drive/Datasets/Pneumonia Chest Xray/val'
test_path = '/content/drive/My Drive/Datasets/Pneumonia Chest Xray/test'

#import the VGG16 architecture and add preprocessing layer, we are using ImageNet weights
VGG16_model = keras.applications.vgg16.VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top = False)

#freeze the weights of the pre-trained layers
for layer in VGG16_model.layers:
  layer.trainable = False
  
#useful for getting number of output classes
folders = glob('/content/drive/My Drive/Datasets/Pneumonia Chest Xray/train/*')

VGG16_model.summary()

#adding our own layers
layer_flatten = keras.layers.Flatten()(VGG16_model.output)
output = keras.layers.Dense(len(folders), activation = "softmax")(layer_flatten)
model = keras.Model(inputs = VGG16_model.input, outputs = output)

#summary of our model
model.summary()

#compile the model and specify loss function and optimizer
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#use the ImageDataGenerator class to load images from the dataset
train_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

valid_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

test_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

#make sure you provide the same target size as initialied for the image size
training_set = train_data_generator.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

validation_set = valid_data_generator.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical')

test_set = test_data_generator.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical')

#train the model
history = model.fit_generator(
  training_set,
  validation_data = validation_set,
  epochs = 5,
  steps_per_epoch = len(training_set),
  validation_steps = len(validation_set)
)

#save the model as an h5 file
model.save('/content/drive/My Drive/Datasets/Pneumonia Chest Xray/model_vgg16.h5')

#plot the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#plot the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#evaluate the model
model.evaluate(test_set)

#using the model to make predictions
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)