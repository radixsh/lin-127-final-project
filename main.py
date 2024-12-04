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

    print("\n\nvars(trans) =")
    pprint(vars(trans))

    print("\n\nvars(trans.utterances[0]) =")
    pprint(vars(trans.utterances[0]))

    return

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

"""
from swda import Transcript
import os

def save_transcripts_for_fasttext(base_dir, metadata_file, output_file):
    with open(output_file, 'w') as out_f:
        # Step 1: Process each subdirectory
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # Step 2: Iterate over .utt.csv files
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
                    for utterance in transcript.utterances:
                        # Extract speaker's education level from metadata
                        caller = utterance.caller  # 'A' or 'B'
                        conversation_no = utterance.conversation_no
                        education_level = transcript.metadata[conversation_no][caller].get("education", "unknown")

                        # Format FastText-compatible line
                        if utterance.text:  # Ensure the utterance has text
                            line = f"__label__{education_level} {utterance.text.lower()}\n"
                            out_f.write(line)

    print(f"Processed all transcripts and saved to {output_file}")
"""

def save_transcripts_for_fasttext(subdirs, metadata_file, output_file):
    with open(output_file, 'w') as out_f:
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
                    print(f"Analyzing {filepath}...")

                    # Step 3: Create a Transcript object
                    try:
                        transcript = Transcript(filepath, metadata_file)
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e}")
                        continue

                    # Step 4: Iterate over utterances in the Transcript
                    for utterance in transcript.utterances:
                        # Extract speaker's education level from metadata
                        caller = utterance.caller  # 'A' or 'B'
                        conversation_no = utterance.conversation_no
                        # education_level = transcript.metadata[conversation_no][caller].get("education", "unknown")
                        education_level = (transcript.metadata[conversation_no]["from_caller_education"]
                                           if caller == "A"
                                           else
                                           transcript.metadata[conversation_no]["to_caller_education"])
                        # transcript.metadata[conversation_no][caller].get("education", "unknown")
                        sex = (transcript.metadata[conversation_no]["from_caller_sex"]
                               if caller == "A"
                               else
                               transcript.metadata[conversation_no]["to_caller_sex"])

                        # Format FastText-compatible line
                        if utterance.text:  # Ensure the utterance has text
                            line = f"__label__{education_level}{sex} {utterance.text.lower()}\n"
                            out_f.write(line)

    print(f"Processed all transcripts and saved to {output_file}")

def main():
    unzip()

    base_dir = "swda"
    metadata_file = os.path.join(base_dir, "swda-metadata.csv")

    # Process contents of training file and output FastText-labeled stuff
    # train_utt = "swda/sw00utt/sw_0001_4325.utt.csv"
    # utt_to_fasttext(train_utt, train_ft, metadata_file)
    train_ft = "train.ft"
    if not os.path.exists(train_ft):
        TRAIN_DIRS = ["sw00utt", "sw01utt"]
        TRAIN_DIRS = [os.path.join(base_dir, filename) for filename in TRAIN_DIRS]
        save_transcripts_for_fasttext(TRAIN_DIRS, metadata_file, train_ft)

    model = fasttext.train_supervised(train_ft)

    # Measure performance on training set
    print(f"Performance on training set: "
          f"\t{model.test('train.ft')[1]*100:.2f}% accuracy")

    model.save_model("trained_model.bin")
    # Retrieve later with:
    # model = fasttext.load_model("trained_model.bin")

    # validation_utt = "swda/sw00utt/sw_0002_4330.utt.csv"
    # utt_to_fasttext(validation_utt, validation_ft, metadata_file)
    validation_ft = "validation.ft"
    if not os.path.exists(validation_ft):
        VALIDATION_DIRS = ["sw02utt", "sw03utt"]
        VALIDATION_DIRS = [os.path.join(base_dir, filename) for filename in VALIDATION_DIRS]
        save_transcripts_for_fasttext(VALIDATION_DIRS, metadata_file, validation_ft)

    # Measure performance on validation set
    print(f"Performance on validation set: "
          f"\t{model.test('validation.ft')[1]*100:.2f}% accuracy")

    # test_utt = "swda/sw00utt/sw_0003_4103.utt.csv"

if __name__ == "__main__":
    main()
