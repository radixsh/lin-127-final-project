import os
import zipfile
import requests
import fasttext
import nltk
import csv

from swda import Transcript

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def unzip():
    zip_filename = "swda.zip"
    subdir = "swda"
    url = "https://github.com/cgpotts/swda/raw/master/swda.zip"

    # Ensure the subdirectory exists:
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    # Ensure the zip exists:
    if not os.path.exists(zip_filename):
        print(f'{zip_filename} not found. Downloading from {url}...')
        response = requests.get(url)
        with open(zip_filename, 'wb') as f:
            f.write(response.content)

    # If 'swda/swda-metadata.csv' does not exist, then this indicates the zip
    # has not been unzipped. So we should unzip it now:
    if not os.path.exists(os.path.join(subdir, 'swda-metadata.csv')):
        # Extract zip file into that subdir
        print(f'Extracting {zip_filename} into {subdir}...')
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall()

# Format .utt into FastText-labeled format
def utt_to_fasttext(input_file, output_file, metadata_filename):
    trans = Transcript(input_file, metadata_filename)

    with open(output_file, 'w', encoding='utf-8') as out:
        for utt in trans.utterances:
            # Tokenize the utterance into sentences
            sentences = nltk.tokenize.sent_tokenize(utt.text)

            # Process each sentence
            for sentence in sentences:
                # Count words in the sentence
                word_count = len(nltk.tokenize.word_tokenize(sentence))

                # Create the formatted FastText line
                formatted = (f'__label__{utt.caller_sex} '
                             f'wordcount:{word_count} '
                             f'tag:{utt.act_tag} '
                             f'sentence:"{sentence}"')

                # Write to the output file
                out.write(formatted.lower() + "\n")

def main():
    unzip()

    metadata_filename = "swda/swda-metadata.csv"

    # For each file, process its contents and output FastText-labeled stuff into
    # another file
    # (For now, just manually set the training filename)
    train_utt = "swda/sw00utt/sw_0001_4325.utt.csv"
    train_ft = "train.ft"
    utt_to_fasttext(train_utt, train_ft, metadata_filename)

    model = fasttext.train_supervised(train_ft)

    # Measure performance on training set
    print(f"Performance on training set: "
          f"\t{model.test('train.ft')[1]*100:.2f}% accuracy")

    validation_utt = "swda/sw00utt/sw_0002_4330.utt.csv"
    validation_ft = "validation.ft"
    utt_to_fasttext(validation_utt, validation_ft, metadata_filename)

    # Measure performance on validation set
    print(f"Performance on validation set: "
          f"\t{model.test('validation.ft')[1]*100:.2f}% accuracy")

    # test_utt = "swda/sw00utt/sw_0003_4103.utt.csv"

if __name__ == "__main__":
    main()
