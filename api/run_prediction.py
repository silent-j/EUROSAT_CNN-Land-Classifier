import os
import argparse
import requests

API_URL = "http://127.0.0.1:5000/predict"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', action='store', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    PATH = args.PATH
    
    if os.path.isfile(PATH):
    
        image = open(PATH, "rb").read()
        payload = {"image": image}
    
        r = requests.post(API_URL, files=payload).json()
        
        if r["success"]:
            for (i, result) in enumerate(r["predictions"]):
                 print("{}. {}: {:.4f}".format(i + 1, result["label"],
                    result["probability"]))
        else:
            print("Failed Request")
    else:
        print("dir-arguments are not supported. PATH must be file")
