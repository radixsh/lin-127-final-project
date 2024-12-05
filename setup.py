import os
import zipfile
import requests
import fasttext
import nltk

from train import train

def get_swda(py_url, zip_url, subdir):
    # Ensure `swda.py` exists
    py_filename = "swda.py"
    if not os.path.exists(py_filename):
        print(f'{py_filename} not found. Downloading from {py_url}...')
        response = requests.get(py_url)
        with open(py_filename, 'wb') as f:
            f.write(response.content)

    # Ensure the zip exists. If 'swda/swda-metadata.csv' does not exist, then
    # this indicates the zip has not been extracted, so unzip it now:
    if not os.path.exists(os.path.join(subdir, 'swda-metadata.csv')):
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        zip_filename = "swda.zip"
        if not os.path.exists(zip_filename):
            print(f'{zip_filename} not found. Downloading from {zip_url}...')
            response = requests.get(zip_url)
            with open(zip_filename, 'wb') as f:
                f.write(response.content)

        # Extract zip file into subdir
        print(f'Extracting {zip_filename} into {subdir}...')
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall()

def setup():
    data_dir = "swda"

    # Download `swda.py` and `swda.zip` if necessary
    py_url = "https://github.com/cgpotts/swda/raw/master/swda.py"
    zip_url = "https://github.com/cgpotts/swda/raw/master/swda.zip"
    get_swda(py_url, zip_url, data_dir)

if __name__ == "__main__":
    setup()
