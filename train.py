import os
import string
import time
from collections import defaultdict
import re
import itertools

import zipfile
import requests
import fasttext
import nltk

from extract_features import format_sentence, format_conv, get_word_lemma_counts, purge_enclosed, clean_square_brackets

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

DATA_DIR = "swda"

wordcounts = defaultdict(float)
male_wordcounts = defaultdict(float)
female_wordcounts = defaultdict(float)

def add_words(output_file, transcript):
    metadata = transcript.metadata
    idx = transcript.conversation_no

    for utt in transcript.utterances:
        is_first_speaker = (utt.caller == 'A')

        sex = (metadata[idx]["from_caller_sex"] if utt.caller == "A"
               else metadata[idx]["to_caller_sex"])

        # Divide utterance into sentences
        tokens = re.split(r'\s+|/(?=\s)|(?<=\s)/|(?<!\w)/|(?<=\s)[^\w\']',
                          utt.text)
        words = [token for token in tokens if token.isalnum()]

        # Process each sentence
        for word in words:
            formatted = (f'__label__{utt.caller_sex} '
                         f'is_first_speaker:{is_first_speaker} '
                         f'type:word '
                         f'word:{word}')
            output_file.write(formatted + '\n')

def add_sentences(output_file, transcript):
    metadata = transcript.metadata
    idx = transcript.conversation_no

    for utt in transcript.utterances:
        is_first_speaker = (utt.caller == 'A')

        sex = (metadata[idx]["from_caller_sex"] if utt.caller == "A"
               else metadata[idx]["to_caller_sex"])

        # Divide utterance into sentences
        sentences = nltk.tokenize.sent_tokenize(utt.text)

        # Process each sentence
        for sentence in sentences:
            sentence_lemmas = get_word_lemma_counts(sentence)

            for lemma, count in sentence_lemmas.items():
                wordcounts[lemma] += count
                if sex == 'MALE':
                    male_wordcounts[lemma] += count
                elif sex == 'FEMALE':
                    female_wordcounts[lemma] += count

            if sentence == "/":
                # Ignore "sentences" that are just "/"
                continue

            is_backchannel = ('bh' == utt.act_tag)

            # Create the formatted FastText line
            # Returns None if the line was garbage, e.g. just "/"
            formatted_sent = format_sentence(sentence)

            if formatted_sent:

                preformat = f'__label__{utt.caller_sex} '
                f'is_first_speaker:{is_first_speaker} '
                f'type:sentence '
                f'is_bh:{is_backchannel} '

                formatted_sent = preformat + formatted_sent
                output_file.write(formatted_sent)

def side_to_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text)

    result = ""
    # Go through and destroy anything too short, and clean up the others
    for sent in sentences:
        if len(sent) <= 8:
            continue
        sent = purge_enclosed(sent)
        sent = " ".join(clean_square_brackets(sent).split())
        sent = sent.lower()
        result += sent + " "

    return result

def add_conversations(output_file, transcript):
    metadata = transcript.metadata
    idx = transcript.conversation_no

    for caller in ["A", "B"]:
        is_first_speaker = (caller == 'A')

        sex = (metadata[idx]["from_caller_sex"] if caller == "A"
               else metadata[idx]["to_caller_sex"])

        side = ""
        for utt in transcript.utterances:
            if utt.caller == caller:
                side += side_to_sentences(utt.text)

        sentences = nltk.tokenize.sent_tokenize(side)
        broken_at_slashes = []
        for sent in sentences:
            broken_at_slashes.extend(sent.split("/"))

        lengths = []
        for sent in broken_at_slashes:
            lengths.append(len(nltk.tokenize.word_tokenize(sent)))

        avg_sentence_length = sum(lengths) / len(lengths)
        if avg_sentence_length < 7:
            avg_length = "short"
        elif avg_sentence_length < 14:
            avg_length = "medium"
        else:
            avg_length = "long"

        # Create the formatted FastText line
        formatted = (f'__label__{sex} '
                     f'is_first_speaker:{is_first_speaker} '
                     f'type:conversation_side '
                     f'avg_length:{avg_length} '
                     f'side:"{side}"')

        # Write to the output file
        output_file.write(formatted.lower() + "\n")

def format_partial_conversation(features, tuple_of_sentences):
    # avg_sentence_length is currently broken because tuple_of_sentences doesn't
    # properly split at "/" but rather only at "." (and whatever else nltk
    # sent_tokenize splits at, in side_to_sentences() call)
    '''
    lengths = []
    for sent in tuple_of_sentences:
        lengths.append(len(nltk.tokenize.word_tokenize(sent)))

    avg_sentence_length = sum(lengths) / len(lengths)
    if avg_sentence_length < 7:
        features['avg_length'] = "short"
    elif avg_sentence_length < 14:
        features['avg_length'] = "medium"
    else:
        features['avg_length'] = "long"
    '''

    # Create the formatted FastText line
    formatted = f"__label__{features['sex']} "
    for feature in features.keys():
        if feature == 'sex':    # Already did this one as __label__ for FastText
            continue
        formatted += f"{feature}:{features[feature]} "
    formatted += f'partial_conversation:"{tuple_of_sentences}"'
    return formatted

def add_partial_conversations(output_file, transcript):
    metadata = transcript.metadata
    idx = transcript.conversation_no

    for caller in ["A", "B"]:
        features = {'is_first_speaker': (caller == 'A'),
                    'sex': (metadata[idx]["from_caller_sex"] if caller == "A"
                            else metadata[idx]["to_caller_sex"])}

        # Get all sentences spoken by caller A
        side = []
        side_count = 0
        for utt in transcript.utterances:
            if utt.caller == caller:
                # print(f'side_to_sentences: {side_to_sentences(utt.text)}')
                # sents = side_to_sentences(utt.text)
                # sentences.extend(side_to_sentences(utt.text))
                side.append(utt)
                side_count += 1
        GRAM_LENGTH = side_count - 1

        # Use itertools.combinations to get every possible combination of 10
        # sentences spoken by caller A
        sentence_ngrams = []
        utterance_strings = [utt.text for utt in side]
        if len(side) < GRAM_LENGTH:
            sentence_ngrams = [utterance_strings]
        else:
            sentence_ngrams = list(itertools.combinations(
                utterance_strings, GRAM_LENGTH))
            # sentence_ngrams = magic(transcript.utterances)

        # Process every combination of GRAM_LENGTH sentences
        for sentence_ngram in sentence_ngrams:
            # print(f'sentence_ngram: {sentence_ngram}')
            # print(f'type(sentence_ngram): {type(sentence_ngram)}')
            # Process and write out
            formatted = format_partial_conversation(features, sentence_ngram)
            output_file.write(formatted + "\n")
        # print(f"Processed all {GRAM_LENGTH}-utt subsets of caller {caller}'s "
        #       f"side of conversation {idx}")

        '''
        sentences = nltk.tokenize.sent_tokenize(side)
        broken_at_slashes = []
        for sent in sentences:
            broken_at_slashes.extend(sent.split("/"))

        lengths = []
        for sent in broken_at_slashes:
            lengths.append(len(nltk.tokenize.word_tokenize(sent)))

        avg_sentence_length = sum(lengths) / len(lengths)
        if avg_sentence_length < 7:
            avg_length = "short"
        elif avg_sentence_length < 14:
            avg_length = "medium"
        else:
            avg_length = "long"
        '''

def make_fasttext(subdirs, output_file, Transcript):
    with open(output_file, 'w') as outfile:
        # Step 1: Process each specified subdirectory
        for subdir_path in subdirs:
            if not os.path.isdir(subdir_path):
                print(f"Skipping invalid directory {subdir_path}")
                continue

            # Step 2: Iterate over .utt.csv files
            print(f"Processing files in {subdir_path}...")
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

                    # add_words(outfile, transcript)
                    # add_conversations(outfile, transcript)
                    add_partial_conversations(outfile, transcript)
                    # add_sentences(outfile, transcript)

    print(f"Saved sentences and conversation sides to {output_file}")

def validate(model, VALIDATION_DIRS, Transcript):
    # Preprocess the validation data
    validation_ft = "validation.ft"
    VALIDATION_DIRS = [os.path.join(DATA_DIR, filename) for filename in VALIDATION_DIRS]
    print(f"VALIDATION_DIRS: {VALIDATION_DIRS}")
    make_fasttext(VALIDATION_DIRS, validation_ft, Transcript)

    # Measure performance on validation set
    validation_performance = model.test('validation.ft')
    print(f"Performance on validation set "
          f"({validation_performance[0]} entries): "
          f"\t{validation_performance[1]*100:.2f}% precision, "
          f"{validation_performance[2]*100:.2f}% recall")

def test(model, TEST_DIRS, Transcript):
    # Preprocess the test data
    test_ft = "test.ft"
    TEST_DIRS = [os.path.join(DATA_DIR, filename) for filename in TEST_DIRS]
    print(f"TEST_DIRS: {TEST_DIRS}")
    make_fasttext(TEST_DIRS, test_ft, Transcript)

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
        if i == 0:#< 7:
            TRAIN_DIRS.append(filename)
        elif i == 1:#< 10:
            VALIDATION_DIRS.append(filename)
        elif i == 2:
            TEST_DIRS.append(filename)

    # The import needs to be in this function, not in the root namespace,
    # because `setup` needs to import train() without knowing what `Transcript`
    # is
    from swda import Transcript

    # Preprocess the training data: Combine files in TRAIN_DIRS into one big
    # training file in FastText format
    train_ft = "train.ft"
    TRAIN_DIRS = [os.path.join(DATA_DIR, filename) for filename in TRAIN_DIRS]
    print(f"TRAIN_DIRS: {TRAIN_DIRS}")
    make_fasttext(TRAIN_DIRS, train_ft, Transcript)

    word_disparities = defaultdict(float)

    print("Total words: " + str(sum(wordcounts.values())))
    print("Unique words: " + str(len(wordcounts.keys())))

    words_filtered = 0

    for lemma, count in wordcounts.items():
        # 72 is about 0.01% of the corpus
        if count > 72:
            disparity = abs(male_wordcounts[lemma] - female_wordcounts[lemma]) / count
            word_disparities[lemma] = disparity
        else:
            words_filtered += 1

    print("Words remaining: " + str(len(word_disparities.keys())))

    print("Words filtered: " + str(words_filtered))

    top_divisive_words = dict(sorted(word_disparities.items(),
                         key=lambda x: x[1],
                         reverse=True)[:30]).keys()
    print("Most divisive words: " + str(top_divisive_words))

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

    # No peeking
    # test(model, TEST_DIRS, Transcript)

    end = time.time()
    print(f"Finished overall in {end - start:.2f} seconds")

if __name__ == "__main__":
    train()
