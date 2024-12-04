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

    # Check if the subdirectory already exists
    if not os.path.exists(subdir):
        os.makedirs(subdir)

        # Extract zip file into that subdir
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(subdir)

# Format .utt into FastText-labeled format
def utt_to_fasttext(input_file, output_file):
    # Open the CSV file and read it
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # Read CSV into a dictionary format
        with open(output_file, 'w', encoding='utf-8') as out:
            # Loop through each row (utterance)
            for row in reader:
                # Extract relevant columns from the row
                act_tag = row['act_tag']
                caller_sex = row['caller']  # Assuming 'caller' indicates sex (e.g., 'A' or 'B')
                text = row['text']

                # Tokenize the text into sentences
                sentences = nltk.tokenize.sent_tokenize(text)

                # Process each sentence
                for sentence in sentences:
                    # Count words in the sentence
                    word_count = len(nltk.tokenize.word_tokenize(sentence))

                    # Create the formatted FastText line
                    formatted = (f'__label__{caller_sex} '
                                 f'wordcount:{word_count} '
                                 f'tag:{act_tag} '
                                 f'sentence:"{sentence}"')

                    # Write to the output file
                    out.write(formatted.lower() + "\n")

def main():
    unzip()

    # For each file, process its contents and output FastText-labeled stuff into
    # another file
    # (For now, just manually set the training filename)
    train_utt = "swda/sw00utt/sw_0001_4325.utt.csv"
    train_ft = "train.ft"
    utt_to_fasttext(train_utt, train_ft)

    model = fasttext.train_supervised(train_ft)

    # Measure performance on training set
    print(f"Performance on training set: "
          f"\t{model.test('train.ft')[1]*100:.2f}% accuracy")

    validation_utt = "swda/sw00utt/sw_0002_4330.utt.csv"
    validation_ft = "validation.ft"
    utt_to_fasttext(validation_utt, validation_ft)

    # Measure performance on validation set
    print(f"Performance on validation set: "
          f"\t{model.test('validation.ft')[1]*100:.2f}% accuracy")

    # test_utt = "swda/sw00utt/sw_0003_4103.utt.csv"

if __name__ == "__main__":
    main()
