# DSND-CapstoneProject-SatelliteImageCNN

The repository is dedicated to an independent data science project, for completion of the Udacity Data Science Nanodegree. This project leverages the application of transfer learning on satellite image data. The task is to successfully predict the type of land coverage in a given satellite photo. 

# I. Requirements & Dependencies:

- Python 3.x

- tensorflow==1.12.0

- keras==2.3.0

- flask==1.0.2

Model training was completed in a GPU-enabled Kaggle kernel.  

# II. Data:
Data is provided publicly by the Deutsches Forschungszentrum für Künstliche Intelligenz (German Research Center for Artificial Intelligence). 

The dataset contains 27,000 64x64p Sentinel-2 Images in RGB mode of various land classifications. The dataset is divided into 10 class labels ranging from natural to urban geographic features. The data can be downloaded, and passing the path to the data's directory as an argument to preprocessing.py will split the dataset into training and testing directories, based on a provided 'test_size' argument. Class size distributions and labels are explored in the jupyter notebook.

# III. Files:

  - repo
  
    - \predict.py: command-line application to predict land cover on an image file
    
    - \preprocessing.py: command-line application to split the dataset directory into training & testing directories
    
    - \train.py: command-line application for training a model
    
    - \utils.py: utility file
    
    - \EUROSAT_NB.ipynb: data exploration, model training and evaluation
    
    - \api
    
      - \app.py: Deploy trained model to API endpoint using Flask

      - \run_prediction.py: command-line application for predicting land cover on an image file

# IV. Usage:
The repo contains command-line applications for training a model, and for predicting land cover in an input image using the trained model. The model can also be deployed as a simple REST API.

## a. Running train.py and predict.py:

1. download the data from http://madm.dfki.de/downloads from under the 'Datasets for Machine Learning' section. Data generator is titled 'EUROSAT (RGB color space images)'.

2. pass the path to the downloaded directory as argument to 'preprocessing.py', and a float representing the proportion of the data to use as a testing set

3. run train.py:

  - [train_dir] str, training set directory
  - [test_dir] str, testing set directory
  - [save_path] str, path to save trained model to
  - [--epochs] default=10, number of epochs for training
  - [--lr] default=0.01, learning rate
  - [--batch_size] default=64, batch size for image generators
  - [--fine_tune] default=True, turns on fine-tuning during the training process
  - [--eval] default=True, evaluate model performance on testing data. Display performance metrics

4. run predict.py:

  - [input_path] str, path to image file for prediction
  - [model_path] str, path to trained model
  
## b. Deploying model as REST API:

1. 

2. 

