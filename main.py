import os
import tarfile
import urllib.request
import fasttext
import nltk

nltk.download('punkt')

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

metadata_filename = "swda/swda-metadata.csv"
# Format it in a way FastText wants
def process_data(input_file, output_file):

    # Make a Transcript obj for this file
    shitty_thing = "swda/sw00utt/sw_0001_4325.utt.csv"
    trans = Transcript(shitty_thing, metadata_filename)

    # with open(target_file, 'r') as f:
    #     lines = f.readlines()

    with open(output_file, 'w') as out:
        for utt in trans.utterances:
            for sentence in nltk.tokenize.sent_tokenize(utt):
                wordcount = len(nltk.tokenize.word_tokenize(sentence))
                formatted_line = (f"__label__{utt.caller_sex} "
                                  f"wordcount:{wc} "
                                  f"sentence:{sentence}")
                out.write(formatted_line.lower() + "\n")


def main():
    # For each file, process its contents and output FastText-labeled stuff into
    # another file

    # For now, just manually setting the filename
    target_file = "swb1_dialogact_annot/sw00utt/sw_0001_4325.utt"
    the_thing_it_becomes = "output.txt"
    process_data(target_file, the_thing_it_becomes)
