# source: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
import tensorflow as tf
import io
from PIL import Image
import numpy as np

from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)
global model, graph

graph = tf.get_default_graph()
model = load_model(r'C:\Users\James\Desktop\EUROSAT\app\vgg16_eurosat.h5')

class_labels = np.load(r'C:\Users\James\Desktop\EUROSAT\class_indices.npy')

def process_image(image, target):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # init data dictionary to be returned
    data = {'success':False}
    
    if request.method == "POST": # image properly uploaded
        if request.files.get("image"):
            
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # preprocess
            image = process_image(image, target=(64, 64))
            
            # predict
            with graph.as_default():
                preds = model.predict(image)
                       
            response = {
                    'pred': {
                            'AnnualCrops': preds[0][0],
                            'Forest': preds[0][1],
                            'HerbacceousVegetation': preds[0][2],
                            'Highway': preds[0][3],
                            'Industrial': preds[0][4],
                            'Pasture': preds[0][5],
                            'PermanentCrop': preds[0][6],
                            'Residential': preds[0][7],
                            'River': preds[0][8],
                            'Sea_Lake': preds[0][9]
                            }
                    }
                    
            data["predictions"] = []
            
            # return predictions
            for (k, v) in response['pred'].items():
                r = {"label": k, "probability":float(v)}
                data["predictions"].append(r)
            
            data["success"] = True
    
    return jsonify(data)


if __name__=="__main__":
    
    print("* Loading Keras model...")
    app.run()