import os
import zipfile
import requests
import fasttext
import nltk

from swda import Transcript

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def download():
    # somethign is weird
    pass

    # url = 'https://github.com/cgpotts/swda/blob/master/swda.zip?raw=true'
    # download_dir = "data"

    # # Check if the subdirectory already exists
    # if not os.path.exists(download_dir):
    #     os.makedirs(download_dir)

    # # Download the zip file
    # response = requests.get(url)
    # with open(zip_file_path, 'wb') as f:
    #     f.write(response.content)

    # # Extract the zip file if not already extracted
    # if not os.path.exists(os.path.join(download_dir, 'swda')):
    #     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    #         zip_ref.extractall(download_dir)

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
            for sentence in nltk.tokenize.sent_tokenize(utt.text):
                formatted = (f'__label__{utt.caller_sex} '
                             f'wordcount:{len(nltk.tokenize.word_tokenize(sentence))} '
                             f'tag:{utt.act_tag} '
                             f'sentence:"{sentence}"')
                out.write(formatted.lower() + "\n")

def main():
    download()

    # For each file, process its contents and output FastText-labeled stuff into
    # another file
    # For now, just manually setting the filename
    target_file = "swb1_dialogact_annot/sw00utt/sw_0001_4325.utt"
    the_thing_it_becomes = "output.txt"
    process_data(target_file, the_thing_it_becomes)

if __name__ == "__main__":
    main()
