# DSND-CapstoneProject-SatelliteImageCNN

The repository is dedicated to an independent data science project, for completion of the Udacity Data Science Nanodegree. This project leverages the application of transfer learning on satellite image data. The task is to successfully predict the type of land coverage in a given satellite photo. 

# Requirements & Dependencies

- Python 3.x

- tensorflow==1.12.0

- keras==2.3.0

- flask==1.0.2

Model training was completed in a GPU-enabled Kaggle kernel.  

# Data
Data is provided publicly by the Deutsches Forschungszentrum für Künstliche Intelligenz (German Research Center for Artificial Intelligence). 

The dataset contains 27,000 64x64p Sentinel-2 Images in RGB mode of various land classifications. The dataset is divided into 10 class labels ranging from natural to urban geographic features. The data can be downloaded, and passing the path to the data's directory as an argument to preprocessing.py will split the dataset into training and testing directories, based on a provided 'test_size' argument. Class size distributions and labels are explored in the jupyter notebook.

# 


