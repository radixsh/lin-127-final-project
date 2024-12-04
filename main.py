import os
import tarfile
import urllib.request

# Download tar and unzip it here if the folder doesn't already exist
foldername = "swb1_dialogact_annot"
if not os.path.exists(foldername):
    filename = "swb1_dialogact_annot.tar.gz"
    url = "http://www.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz"
    # Check if the file exists in the current directory
    if not os.path.exists(filename):
        print(f"{filename} not found in the current directory. Downloading...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}.")
    else:
        print(f"{filename} already exists in the current directory.")

    # Extract the file
    if tarfile.is_tarfile(filename):
        print(f"Extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=foldername)  # Extracts to the current directory
        print(f"Extraction complete.")
    else:
        print(f"{filename} is not a valid tar.gz file.")

# Iterate through it

