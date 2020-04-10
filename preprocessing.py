import re
import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

def parse_args():
    '''
    command line argument parser
        - data_dir (path): path to dataset containing images in class subdirectories
        - test_size (float): proportion of data to seperate into 'testing' directory
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action='store', type=str)
    parser.add_argument('test_size', action='store', type=float) 
    return parser.parse_args()

def main(x, y, test_size):
    '''
    split directory of class subdirectories using StratifiedShuffleSplit
        - x (Pandas series): series of paths to images in subdirs of data_dir
        - y (list or array-like): class labels for paths in x
        - test_size (float): proportion of data to seperate into 'testing' directory
    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=69)
    
    
    for train_idx, test_idx in split.split(X, y):
        
        train_paths = list(x.loc[train_idx])
        test_paths = list(x.loc[test_idx])
    
        new_train_paths = [os.path.join(TRAIN_DIR, i.split('_')[0], i) for i in train_paths]
        new_test_paths = [os.path.join(TEST_DIR, i.split('_')[0], i) for i in test_paths]
    
        train_paths = [os.path.join(DATASET, i.split('_')[0], i) for i in train_paths]
        test_paths = [os.path.join(DATASET, i.split('_')[0], i) for i in test_paths]
    
        train_path_map = list((zip(train_paths, new_train_paths)))
        test_path_map = list((zip(test_paths, new_test_paths)))
        
        print("moving training data...")
        for i in tqdm(train_path_map):
            if not os.path.exists(i[1]):
                if not os.path.exists(re.sub('training', 'testing', i[1])):
                    shutil.move(i[0], i[1])
        
        print("moving testing data...")
        for i in tqdm(test_path_map):
            if not os.path.exists(i[1]):
                if not os.path.exists(re.sub('training', 'testing', i[1])):
                    shutil.move(i[0], i[1])
        

if __name__=="__main__":
    
    args = parse_args()
    DATASET = args.data_dir
    TEST_SIZE = args.test_size
    TRAIN_DIR = os.path.join(DATASET, 'training')
    TEST_DIR = os.path.join(DATASET, 'testing')   
    NUM_CLASSES=len(os.listdir(DATASET))
    LABELS = os.listdir(DATASET)
    
    for path in (TRAIN_DIR, TEST_DIR):
        if not os.path.exists(path):
            os.mkdir(path)
    
    print("moving files..")
    for l in LABELS:
        
        if not os.path.exists(os.path.join(TRAIN_DIR, l)):
            os.mkdir(os.path.join(TRAIN_DIR, l))
    
        if not os.path.exists(os.path.join(TEST_DIR, l)):
            os.mkdir(os.path.join(TEST_DIR, l))
            
    data = {}
    
    for l in LABELS:
        for img in os.listdir(os.path.join(DATASET,l)):
            data.update({img: l})
    
    X = pd.Series(list(data.keys())) # paths need to be stored in Pandas series
    y = pd.get_dummies(pd.Series(data.values()))
    
    main(X, y, TEST_SIZE)
    
    for l in LABELS:
        if len(os.listdir(os.path.join(DATASET,l))) == 0:
            os.rmdir(os.path.join(DATASET,l))
