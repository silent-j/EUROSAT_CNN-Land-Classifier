import PIL
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', action='store', type=str)
    parser.add_argument('model_path', action='store', type=str)
    
    return parser.parse_args()

def process_input(input_path):    
    img = image.load_img(input_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array/255.0
    img_array = img_array.reshape(1, 64, 64, 3)
    
    return img_array

def predict(img_array):
    probs = model.predict(img_array)
    pred_class_index = np.argmax(probs, axis=1)    
    class_indices = r'..\..\class_indices.npy'
    class_indices = np.load(class_indices).item()
    pred_class_label = [k for k in class_indices.keys() if class_indices[k] == pred_class_index]
    pred_prob = probs.flatten()[pred_class_index].item()    
    
    return pred_class_label[0], pred_prob


def predict_classes(img_array):
    
    probs = model.predict(img_array)
    probs = [np.round(p, 5) for p in probs.flatten()]
    class_indices = r'..\..\class_indices.npy'
    class_indices = np.load(class_indices).item()
    class_probs = dict(zip(class_indices.keys(), probs))
    
    return class_probs

def plot_prediction(input_path, class_label, class_prob):
    
    img = PIL.Image.open(input_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.suptitle(f"Land Coverage: {class_label}", fontsize=15)
    plt.title("Certainty: {}%".format(np.round(class_prob, 2)*100), fontsize=15)
    plt.show(block=True);


if __name__=="__main__":
    
    args = parse_args()
    
    # TODO load model    
    model = load_model(args.model_path)
    
    # TODO predict
    print("processing input...")
    img_array = process_input(args.input_path)
    
    print("predicting land cover...")
    class_label, class_prob = predict(img_array)
    
    # TODO display prediction
    plot_prediction(args.input_path, class_label, class_prob)
    
