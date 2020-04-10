import re
import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, confusion_matrix

########## training functions ##########

def load_generators(train_dir, batch_size, val_split=0.3, class_mode='categorical'):
    '''
    create image generators for training data with validation set. The provided
    directory for train_dir must contain images in their respective class label
    subdir.
        - train_dir (path): path to training data directory containing subdirs
        - batch_size (int): batch size for loading
        - val_split (float): default=0.3; Float to represent proportion of data
        to include in validation set
        - class_mode (str): default='categorical'; Class mode for image generator.
    Returns:
        - training and validation generators
    '''
    train_gen = ImageDataGenerator(
        rescale=1.0/255.0, rotation_range=60, width_shift_range=0.3,
        height_shift_range=0.3, shear_range=0.3, zoom_range=0.3,
        horizontal_flip=True, validation_split=val_split)

    train_generator = train_gen.flow_from_directory(
        directory=train_dir, target_size=(64, 64), batch_size=batch_size,
        class_mode=class_mode, subset='training', color_mode='rgb',
        shuffle=True, seed=69)

    valid_generator = train_gen.flow_from_directory(
        directory=train_dir, target_size=(64, 64), batch_size=batch_size,
        class_mode=class_mode, subset='validation', color_mode='rgb',
        shuffle=True, seed=69)
 
    return train_generator, valid_generator

def compile_model(input_shape, n_classes, optimizer, fine_tune=None):
    '''
    compile a keras model with the pre-trained VGG16 convolutional base
        - input_shape (tuple): shape of input images 
        - n_classes (int): expected num class labels in the output layer
        - optimizer (keras optimier): A keras optimizer 
        - fine_tune (int): default=None; int to select layers from VGG16 convolutional
        base for unfreezing. Layers will be selected through list slicing 'base.layers[int: ]'
    '''
    conv_base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape,
                     pooling='avg')
    
    top_model = conv_base.output
    top_model = Dense(2048, activation='relu')(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    model = Model(inputs=conv_base.input, outputs=output_layer)
        
    if type(fine_tune) == int:
        for layer in conv_base.layers[fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                 metrics=['categorical_accuracy'])    
    return model

def plot_history(history):
    '''
    plot training accuracy and loss across epochs
    '''
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show();

########## evaluation functions ##########
    
def display_results(y_true, y_preds, class_labels):
    
    results = pd.DataFrame(precision_recall_fscore_support(y_true, y_preds),
                          columns=class_labels).T
    results.rename(columns={0: 'Precision',
                           1: 'Recall',
                           2: 'F-Score',
                           3: 'Support'}, inplace=True)
    
    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_preds), 
                            columns=class_labels,
                            index=class_labels)    
    f2 = fbeta_score(y_true, y_preds, beta=2, average='micro')
    print(f"Global F2 Score: {f2}")    
    return results, conf_mat

def plot_predictions(y_true, y_preds, test_generator, class_indices):

    fig = plt.figure(figsize=(20, 10))
    for i, idx in enumerate(np.random.choice(test_generator.samples, size=20, replace=False)):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(test_generator[idx]))
        pred_idx = np.argmax(y_preds[idx])
        true_idx = y_true[idx]              
        plt.tight_layout()
        ax.set_title("{}\n({})".format(class_indices[pred_idx], class_indices[true_idx]),
                     color=("green" if pred_idx == true_idx else "red"))
        
        plt.show(block=True);
