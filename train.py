import os
import string
import time

import zipfile
import requests
import fasttext
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

DATA_DIR = "swda"

def cleanup(sentence):
    # Remove <<sound environment comments>>
    # re.sub(r'<<.*?>>', '', sentence)
    while '<<' in sentence and '>>' in sentence:
        start = sentence.find('<<')
        end = sentence.find('>>', start)
        if start != -1 and end != -1:
            # Remove <<...>> including brackets
            sentence = sentence[:start] + sentence[end + 2:]
        else:
            break

    # Make lowercase
    sentence = sentence.lower()

    # Remove all punctuation
    sentence.translate(str.maketrans('', '', string.punctuation))

    return sentence

def transcripts_to_fasttext(subdirs, output_file, Transcript):
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
                        metadata_file = os.path.join(DATA_DIR, "swda-metadata.csv")
                        transcript = Transcript(filepath, metadata_file)
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e}")
                        continue

                    # Step 4: Iterate over utterances in the Transcript
                    for utt in transcript.utterances:
                        # Extract speaker's education level from metadata
                        caller = utt.caller  # 'A' or 'B'
                        conversation_no = utt.conversation_no
                        # education_level = (transcript.metadata[conversation_no]["from_caller_education"]
                        #                    if caller == "A"
                        #                    else
                        #                    transcript.metadata[conversation_no]["to_caller_education"])
                        sex = (transcript.metadata[conversation_no]["from_caller_sex"]
                               if caller == "A"
                               else
                               transcript.metadata[conversation_no]["to_caller_sex"])

                        is_first_speaker = (utt.caller == 'A')

                        # Divide utterance into sentences
                        sentences = nltk.tokenize.sent_tokenize(utt.text)

                        # Process each sentence
                        for sentence in sentences:
                            if sentence == "/":
                                # Ignore "sentences" that are just "/"
                                continue

                            sentence = cleanup(sentence)

                            # Count words in the sentence
                            word_count = len(nltk.tokenize.word_tokenize(sentence))

                            # Create the formatted FastText line
                            formatted = (f'__label__'
                                         # f'{education_level}'
                                         f'{utt.caller_sex} '
                                         f'is_first_speaker:{is_first_speaker} '
                                         f'wordcount:{word_count} '
                                         f'tag:{utt.act_tag} '
                                         f'sentence:"{sentence.lower()}"')

                            # Write to the output file
                            outfile.write(formatted.lower() + "\n")

    print(f"Processed all transcripts and saved to {output_file}")


def validate(model, VALIDATION_DIRS, Transcript):
    # Preprocess the validation data
    validation_ft = "validation.ft"
    if not os.path.exists(validation_ft):
        VALIDATION_DIRS = [os.path.join(DATA_DIR, filename) for filename in VALIDATION_DIRS]
        transcripts_to_fasttext(VALIDATION_DIRS, validation_ft, Transcript)

    # Measure performance on validation set
    validation_performance = model.test('validation.ft')
    print(f"Performance on validation set "
          f"({validation_performance[0]} entries): "
          f"\t{validation_performance[1]*100:.2f}% precision, "
          f"{validation_performance[2]*100:.2f}% recall")

def test(model, TEST_DIRS, Transcript):
    # Preprocess the test data
    test_ft = "test.ft"
    if not os.path.exists(test_ft):
        TEST_DIRS = [os.path.join(DATA_DIR, filename) for filename in TEST_DIRS]
        transcripts_to_fasttext(TEST_DIRS, test_ft, Transcript)

    # Measure performance on test set
    test_performance = model.test('test.ft')
    print(f"Performance on test set "
          f"({test_performance[0]} entries): "
          f"\t{test_performance[1]*100:.2f}% precision, "
          f"{test_performance[2]*100:.2f}% recall")



def train():
    start = time.time()

    TRAIN_DIRS = []
    VALIDATION_DIRS = []
    TEST_DIRS = []
    for i in range(0, 13):
        filename = f"sw{i:02}utt"
        if i < 7:
            TRAIN_DIRS.append(filename)
        elif i < 10:
            VALIDATION_DIRS.append(filename)
        else:
            TEST_DIRS.append(filename)
    print(f"TRAIN_DIRS: {TRAIN_DIRS}")
    print(f"VALIDATION_DIRS: {VALIDATION_DIRS}")
    print(f"TEST_DIRS: {TEST_DIRS}")

    # The import needs to be in this function, not in the root namespace,
    # because `setup` needs to import train() without knowing what `Transcript`
    # is
    from swda import Transcript

    # Preprocess the training data: Combine files in TRAIN_DIRS into one big
    # training file in FastText format
    train_ft = "train.ft"
    if not os.path.exists(train_ft):
        TRAIN_DIRS = [os.path.join(DATA_DIR, filename) for filename in TRAIN_DIRS]
        transcripts_to_fasttext(TRAIN_DIRS, train_ft, Transcript)

    # Train and test the model on training set
    model = fasttext.train_supervised(train_ft,
                                      # lr=1.0,
                                      epoch=10,
                                      )
    train_performance = model.test('train.ft')
    print(f"Performance on train set "
          f"({train_performance[0]} entries): "
          f"\t{train_performance[1]*100:.2f}% precision, "
          f"{train_performance[2]*100:.2f}% recall")

    model.save_model("model.bin")

    trained = time.time()
    print(f"Finished training in {trained - start:.2f} seconds")

    validate(model, VALIDATION_DIRS, Transcript)
    test(model, TEST_DIRS, Transcript)

    end = time.time()
    print(f"Finished overall in {end - start:.2f} seconds")

if __name__ == "__main__":
    train()
