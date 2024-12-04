import os
import zipfile
import requests
import fasttext
import nltk
import csv
from pprint import pprint

from swda import Transcript

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def unzip(url, subdir):
    # Ensure the subdirectory exists
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    # If 'swda/swda-metadata.csv' does not exist, then this indicates the zip
    # has not been extracted, so unzip it now:
    if not os.path.exists(os.path.join(subdir, 'swda-metadata.csv')):
        # Ensure the zip exists
        if not os.path.exists(f"{subdir}.zip"):
            print(f'{zip_filename} not found. Downloading from {url}...')
            response = requests.get(url)
            with open(zip_filename, 'wb') as f:
                f.write(response.content)

        # Extract zip file into subdir
        print(f'Extracting {zip_filename} into {subdir}...')
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall()

def transcripts_to_fasttext(subdirs, metadata_file, output_file):
    with open(output_file, 'w') as outfile:
        # Step 1: Process each specified subdirectory
        for subdir_path in subdirs:
            if not os.path.isdir(subdir_path):
                print(f"Skipping invalid directory {subdir_path}")
                continue

            # Step 2: Iterate over .utt.csv files
            print(f"Combining/processing files in {subdir_path} into {output_file}...")
            for filename in os.listdir(subdir_path):
                if filename.endswith(".utt.csv"):
                    filepath = os.path.join(subdir_path, filename)

                    # Step 3: Create a Transcript object
                    try:
                        transcript = Transcript(filepath, metadata_file)
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e}")
                        continue

                    # Step 4: Iterate over utterances in the Transcript
                    for utt in transcript.utterances:
                        # Extract speaker's education level from metadata
                        caller = utt.caller  # 'A' or 'B'
                        conversation_no = utt.conversation_no
                        education_level = (transcript.metadata[conversation_no]["from_caller_education"]
                                           if caller == "A"
                                           else
                                           transcript.metadata[conversation_no]["to_caller_education"])
                        sex = (transcript.metadata[conversation_no]["from_caller_sex"]
                               if caller == "A"
                               else
                               transcript.metadata[conversation_no]["to_caller_sex"])

                        # Divide utterance into sentences
                        sentences = nltk.tokenize.sent_tokenize(utt.text)

                        # Process each sentence
                        for sentence in sentences:
                            # Count words in the sentence
                            word_count = len(nltk.tokenize.word_tokenize(sentence))

                            # Create the formatted FastText line
                            formatted = (f'__label__{education_level}{utt.caller_sex} '
                                         f'wordcount:{word_count} '
                                         f'tag:{utt.act_tag} '
                                         f'sentence:"{sentence.lower()}"')

                            # Write to the output file
                            outfile.write(formatted.lower() + "\n")

    print(f"Processed all transcripts and saved to {output_file}")

def main():
    data_dir = "swda"

    # Download/extract swda.zip if necessary
    url = "https://github.com/cgpotts/swda/raw/master/swda.zip"
    unzip(url, data_dir)

    # Preprocess the training data: Combine files in TRAIN_DIRS into one big
    # training file in FastText format
    metadata_file = os.path.join(data_dir, "swda-metadata.csv")
    train_ft = "train.ft"
    if not os.path.exists(train_ft):
        TRAIN_DIRS = ["sw00utt", "sw01utt"]
        TRAIN_DIRS = [os.path.join(data_dir, filename) for filename in TRAIN_DIRS]
        transcripts_to_fasttext(TRAIN_DIRS, metadata_file, train_ft)

    # Train and test the model on training set
    model = fasttext.train_supervised(train_ft,
                                      # lr=1.0,
                                      epoch=100,
                                      )
    train_performance = model.test('train.ft')
    print(f"Performance on train set "
          f"({train_performance[0]} entries): "
          f"\t{train_performance[1]*100:.2f}% precision, "
          f"{train_performance[2]*100:.2f}% recall")

    model.save_model("model.bin")

    # Preprocess the validation data
    validation_ft = "validation.ft"
    if not os.path.exists(validation_ft):
        VALIDATION_DIRS = ["sw02utt", "sw03utt"]
        VALIDATION_DIRS = [os.path.join(data_dir, filename) for filename in VALIDATION_DIRS]
        transcripts_to_fasttext(VALIDATION_DIRS, metadata_file, validation_ft)

    # Measure performance on validation set
    validation_performance = model.test('validation.ft')
    print(f"Performance on validation set "
          f"({validation_performance[0]} entries): "
          f"\t{validation_performance[1]*100:.2f}% precision, "
          f"{validation_performance[2]*100:.2f}% recall")

    # Test later??
    # test_utt = "swda/sw00utt/sw_0003_4103.utt.csv"

if __name__ == "__main__":
    main()
