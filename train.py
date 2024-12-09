import os
import string
import time
from collections import defaultdict
import re
import itertools
import random
from functools import reduce
import statistics

import zipfile
import requests
import fasttext
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

from extract_features import get_word_lemma_counts, purge_enclosed, clean_square_brackets

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

DATA_DIR = "swda"

OVERSAMPLE_FACTOR_AND_EPOCHS = 3

regex_tokenizer = RegexpTokenizer(r'\w+')
syllable_tokenizer = SyllableTokenizer()
lemmatizer = WordNetLemmatizer()

TRAIN_MALE_ENTRIES = 0
TRAIN_FEMALE_ENTRIES = 0
TRAIN_DISCARDED_ENTRIES = 0
VAL_MALE_ENTRIES = 0
VAL_FEMALE_ENTRIES = 0
VAL_DISCARDED_ENTRIES = 0


wordcounts = defaultdict(float)
male_wordcounts = defaultdict(float)
female_wordcounts = defaultdict(float)

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if unknown

def get_word_lemma_counts(sent):
    out = defaultdict(float)

    # Get rid of all annotations, make lowercase, and nuke punctuation
    sent = purge_enclosed(sent)
    sent = " ".join(clean_square_brackets(sent).split())
    sent = sent.lower()
    sent = sent.translate(str.maketrans('', '', string.punctuation))

    words = nltk.word_tokenize(sent)

    for word in words:
        pos = get_wordnet_pos(word)
        lemma = lemmatizer.lemmatize(word, pos=pos)
        out[f'{lemma}'] += 1
    
    return out

# def add_words(output_file, transcript):
#     metadata = transcript.metadata
#     idx = transcript.conversation_no

#     for utt in transcript.utterances:
#         is_first_speaker = (utt.caller == 'A')

#         sex = (metadata[idx]["from_caller_sex"] if utt.caller == "A"
#                else metadata[idx]["to_caller_sex"])

#         # Divide utterance into sentences
#         tokens = re.split(r'\s+|/(?=\s)|(?<=\s)/|(?<!\w)/|(?<=\s)[^\w\']',
#                           utt.text)
#         words = [token for token in tokens if token.isalnum()]

#         # Process each sentence
#         for word in words:
#             formatted = (f'__label__{utt.caller_sex} '
#                          f'is_first_speaker:{is_first_speaker} '
#                          f'type:word '
#                          f'word:{word}')
#             output_file.write(formatted + '\n')

# def add_sentences(output_file, transcript):
#     metadata = transcript.metadata
#     idx = transcript.conversation_no

#     for utt in transcript.utterances:
#         is_first_speaker = (utt.caller == 'A')

#         sex = (metadata[idx]["from_caller_sex"] if utt.caller == "A"
#                else metadata[idx]["to_caller_sex"])

#         # Divide utterance into sentences
#         sentences = nltk.tokenize.sent_tokenize(utt.text)

#         # Process each sentence
#         for sentence in sentences:
#             sentence_lemmas = get_word_lemma_counts(sentence)

#             for lemma, count in sentence_lemmas.items():
#                 wordcounts[lemma] += count
#                 if sex == 'MALE':
#                     male_wordcounts[lemma] += count
#                 elif sex == 'FEMALE':
#                     female_wordcounts[lemma] += count

#             if sentence == "/":
#                 # Ignore "sentences" that are just "/"
#                 continue

#             is_backchannel = ('bh' == utt.act_tag)

#             # Create the formatted FastText line
#             # Returns None if the line was garbage, e.g. just "/"
#             formatted_sent = format_sentence(sentence)

#             if formatted_sent:

#                 preformat = f'__label__{utt.caller_sex} '
#                 #f'is_first_speaker:{is_first_speaker} '
#                 #f'type:sentence '
#                 #f'is_bh:{is_backchannel} '

#                 formatted_sent = preformat + formatted_sent
#                 output_file.write(formatted_sent)

# def add_conversations(output_file, transcript):
#     metadata = transcript.metadata
#     idx = transcript.conversation_no

#     for caller in ["A", "B"]:
#         is_first_speaker = (caller == 'A')

#         sex = (metadata[idx]["from_caller_sex"] if caller == "A"
#                else metadata[idx]["to_caller_sex"])

#         side = ""
#         for utt in transcript.utterances:
#             if utt.caller == caller:
#                 side += side_to_sentences(utt.text)

#         sentences = nltk.tokenize.sent_tokenize(side)
#         broken_at_slashes = []
#         for sent in sentences:
#             broken_at_slashes.extend(sent.split("/"))

#         lengths = []
#         for sent in broken_at_slashes:
#             lengths.append(len(nltk.tokenize.word_tokenize(sent)))

#         avg_sentence_length = sum(lengths) / len(lengths)
#         if avg_sentence_length < 7:
#             avg_length = "short"
#         elif avg_sentence_length < 14:
#             avg_length = "medium"
#         else:
#             avg_length = "long"

#         # Create the formatted FastText line
#         formatted = (f'__label__{sex} '
#                      f'is_first_speaker:{is_first_speaker} '
#                      f'type:conversation_side '
#                      f'avg_length:{avg_length} '
#                      f'side:"{side}"')

#         # Write to the output file
#         output_file.write(formatted.lower() + "\n")

def side_to_sentences(text):
    # Convert tuple into paragraph
    text = ''.join(text)

    # Get sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    # nltk doesn't recognize "/" as a sentence divider, so divide it manually
    split_at_slashes = []
    for sent in sentences:
        split_at_slashes.extend(sent.split("/"))

    result = ""
    # Go through and destroy anything too short, and clean up the others
    for sent in split_at_slashes:
        if len(sent) <= 8:
            continue
        sent = purge_enclosed(sent)
        sent = " ".join(clean_square_brackets(sent).split())
        sent = sent.lower()
        result += sent + " "
    
    sent = sent.strip('#')

    return result, len(split_at_slashes)

DIVISIVE_WORDS = ['husband', 'wonderful', 'wear', 'dress', 'wife', 'huhuh', 'goodness',
                  'dinner', 'tax', 'cook', 'woman', 'gosh', 'color', 'girl', 'god', 
                  'neat', 'mother', 'oh', 'vacation', 'flower', 'told', 'men', 'fish', 
                  'ours', 'university', 'love',]

def format_partial_conversation(features, tuple_of_sentences):
    # Get average length of sentences
    sentences, num_sentences = side_to_sentences(tuple_of_sentences)
    # lengths = []
    # for sent in sentences:
    #     lengths.append(len(nltk.tokenize.word_tokenize(sent)))

    
    
    tokens = nltk.word_tokenize(sentences)
    nontrivial_tokens = [token for token in tokens if len(token) > 1]

    lemmata = []

    for word in tokens:
        pos = get_wordnet_pos(word)
        lemmata.append(lemmatizer.lemmatize(word, pos=pos))

    # Very effective: 55.14%
    # for word in DIVISIVE_WORDS:
    #     if word in lemmata:
    #         features[f'has_{word}'] = "True"
    #     else:
    #         features[f'has_{word}'] = "False"


    # Not very effective: 50.27%
    # features['mentions_children'] = reduce(lambda a, b: a or b, 
    #                            [word in lemmata for word in ["child",
    #                                                       "kid",
    #                                                       "son",
    #                                                       "daughter",
    #                                                       "baby",
    #                                                       ]])

    # avg_sentence_length = len(tokens) / num_sentences
    # if avg_sentence_length < 7:
    #     features['avg_length'] = "short"
    # elif avg_sentence_length < 14:
    #     features['avg_length'] = "medium"
    # else:
    #     features['avg_length'] = "long"


    ### Pronoun features
    # # Count pronouns, loop is faster than .count here
    # num_prons, fp_prons, sp_prons = 0,0,0 
    # male_tp_prons, female_tp_prons, neutral_tp_prons = 0,0,0
    # for word in tokens:
    #     if word == 'i':
    #         fp_prons += 1
    #         num_prons += 1
    #     elif word == 'you':
    #         sp_prons += 1
    #         num_prons += 1
    #     elif word in ['he', 'him', 'his']:
    #         male_tp_prons += 1
    #         num_prons += 1
    #     elif word in ['she', 'her', 'hers']:
    #         female_tp_prons += 1
    #         num_prons += 1
    #     elif word in ['they', 'them', 'theirs']:
    #         neutral_tp_prons += 1
    #         num_prons += 1

    # tp_prons = male_tp_prons + female_tp_prons + neutral_tp_prons

    # features['primary_fp_vs_sp_prons'] = "na"
    # features['primary_person_prons'] = "na"
    # features['primary_gender_prons'] = "na"

    # # No effect on accuracy
    # if fp_prons > sp_prons:
    #     features['primary_fp_vs_sp_prons'] = "fp"
    #     if fp_prons > tp_prons:
    #         features['primary_person_prons'] = "fp"
    #     elif tp_prons > fp_prons:
    #         features['primary_person_prons'] = "tp"
    # elif sp_prons > fp_prons:
    #     features['primary_fp_vs_sp_prons'] = "sp"
    #     if sp_prons > tp_prons:
    #         features['primary_person_prons'] = "sp"
    #     elif tp_prons > sp_prons:
    #         features['primary_person_prons'] = "tp"

    # # Zero effect on accuracy
    # if neutral_tp_prons > male_tp_prons:
    #     if neutral_tp_prons > female_tp_prons:
    #         features['primary_gender_prons'] = "neutral"
    #     elif female_tp_prons > neutral_tp_prons:
    #         features['primary_gender_prons'] = "female"
    # elif male_tp_prons > neutral_tp_prons:
    #     if male_tp_prons > female_tp_prons:
    #         features['primary_gender_prons'] = "male"
    #     elif female_tp_prons > male_tp_prons:
    #         features['primary_gender_prons'] = "female"

    # # Analyze word lengths slightly more than naively
    word_lens = list(map(len, nontrivial_tokens))

    # # This should account for skewed data where someone uses lots of small words
    if len(word_lens) > 0:
        avg_word_length = statistics.mean(word_lens) / num_sentences
    else:
        return None
    
    ## Zero change to accuracy
    # if avg_word_length >= 6:
    #     features['avg_word_length'] = "long"
    # elif avg_word_length >= 3:
    #     features['avg_word_length'] = "medium"
    # else:
    #     features['avg_word_length'] = "short"


    ## Zero change to accuracy
    ## This should be more representative of medium-length words
    nontrivial_word_lens = [n / num_sentences for n in word_lens if n > 3]
    # if len(nontrivial_word_lens) > 0:
    #     median_nontrivial_word_length = statistics.median(nontrivial_word_lens)
    # else:
    #     median_nontrivial_word_length = 0

    # if median_nontrivial_word_length >= 7:
    #     features['median_nontrivial_word_length'] = "long"
    # elif median_nontrivial_word_length >= 5:
    #     features['median_nontrivial_word_length'] = "medium"
    # else:
    #     features['median_nontrivial_word_length'] = "short"


    # about ~16% of english words are 8 or longer characters

    # features['has_long_word_by_chars'] = "False"
    # for n in nontrivial_word_lens:
    #     if n >= 8:
    #         features['has_long_word_by_chars'] = "True"
    #         break

    # Analyze syllables, turn each word into the number of syllables it contains
    # This has some small issues with apostrophes, since we split on them before. 
    syllable_counts = [len(syllable_tokenizer.tokenize(word)) for word in tokens]
    
    avg_sent_syllables = sum(syllable_counts) / num_sentences

    # No change to accuracy
    # if avg_sent_syllables > 30:
    #     features['avg_sent_syllables'] = "many"
    # elif avg_sent_syllables > 15:
    #     features['avg_sent_syllables'] = "medium"
    # else:
    #     features['avg_sent_syllables'] = "few"

    # if len(syllable_counts) > 0:
    #     avg_num_syllables = statistics.mean(syllable_counts)
    # else:
    #     return None

    # nontrivial_syllable_counts = [n for n in syllable_counts if n > 1]

    # if len(nontrivial_syllable_counts) > 0:
    #     median_nontrivial_syllables = statistics.median(nontrivial_syllable_counts)
    # else:
    #     median_nontrivial_syllables = 0

    # if median_nontrivial_syllables > 6:
    #     features['median_nontrivial_syllables'] = "many"
    # elif median_nontrivial_syllables >= 3:
    #     features['median_nontrivial_syllables'] = "medium"
    # else:
    #     features['median_nontrivial_syllables'] = "few"

    # features['has_trisyllabic_or_longer'] = "False"
    # features['has_really_long'] = "False"

    # for n in syllable_counts:
    #     if n >= 4:
    #         features['has_trisyllabic_or_longer'] = "True"
    #         features['has_really_long'] = "True"
    #         break
    #     elif n >= 3:
    #         features['has_trisyllabic_or_longer'] = "True"

    # Create the formatted FastText line
    formatted = f"__label__{features['sex']} "
    for feature in features.keys():
        if feature == 'sex':    # Already did this one as __label__ for FastText
            continue
        formatted += f"{feature}:{features[feature]} "
    #formatted += f'partial_conversation:"{tuple_of_sentences}"'
    return formatted

def add_partial_conversations(output_file, transcript, mode):
    metadata = transcript.metadata
    idx = transcript.conversation_no

    for caller in ["A", "B"]:
        features = {'sex': (metadata[idx]["from_caller_sex"] if caller == "A"
                            else metadata[idx]["to_caller_sex"])}

        # Get all sentences spoken by caller A
        side = []
        for utt in transcript.utterances:
            if utt.caller == caller:
                side.append(utt)
        GRAM_LENGTH = 6

        # Use itertools.combinations to get every possible combination of 10
        # sentences spoken by caller A
        sentence_ngrams = []
        utterance_strings = [utt.text for utt in side]
        if len(side) < GRAM_LENGTH:
            sentence_ngrams = [utterance_strings]
        else:
            sentence_ngrams = [random.sample(utterance_strings, GRAM_LENGTH) for _ in range(int(3 * (len(side) // GRAM_LENGTH)))]
        # elif mode == 'TEST':
        #     # Take duples of sentences
        #     sentence_ngrams = list(itertools.combinations(
        #         utterance_strings, 2))
        
            
        # Pre-count total number of each sex's entry
        if mode == 'TRAIN':
            global TRAIN_MALE_ENTRIES
            global TRAIN_FEMALE_ENTRIES
            global TRAIN_DISCARDED_ENTRIES
            for sentence_ngram in sentence_ngrams:
                if features['sex'] == 'FEMALE':
                    TRAIN_FEMALE_ENTRIES += 1
                elif features['sex'] == 'MALE':
                    TRAIN_MALE_ENTRIES += 1
        elif mode == 'VAL':
            global VAL_MALE_ENTRIES
            global VAL_FEMALE_ENTRIES
            global VAL_DISCARDED_ENTRIES
            for sentence_ngram in sentence_ngrams:
                if features['sex'] == 'FEMALE':
                    VAL_FEMALE_ENTRIES += 1
                elif features['sex'] == 'MALE':
                    VAL_MALE_ENTRIES += 1


        # Manually calculated: 1 - F/M, where F and M are the total numbers of
        # sentence ngrams, respectively
        TRAIN_MOST_COMMON_SEX = 'FEMALE'
        VAL_MOST_COMMON_SEX = 'FEMALE'
        TRAIN_DISCARD_PROBABILITY = 1 - (10910.0 / 14310.0)
        VAL_DISCARD_PROBABILITY = 1 - (4685 / 6987)

        # Process every combination of GRAM_LENGTH sentences
        for sentence_ngram in sentence_ngrams:
            # Discard all excess female entries to balance training data
            if mode == "TRAIN" and features["sex"] == TRAIN_MOST_COMMON_SEX:
                if np.random.random() < TRAIN_DISCARD_PROBABILITY:
                    TRAIN_DISCARDED_ENTRIES += 1
                    continue
            if mode == "VAL" and features["sex"] == VAL_MOST_COMMON_SEX:
                if np.random.random() < VAL_DISCARD_PROBABILITY:
                    VAL_DISCARDED_ENTRIES += 1
                    continue

            formatted = format_partial_conversation(features, sentence_ngram)
            if formatted:
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

def make_fasttext(subdirs, output_file, Transcript, mode):
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
                    add_partial_conversations(outfile, transcript, mode = mode)
                    # add_sentences(outfile, transcript)

    print(f"Saved sentences and conversation sides to {output_file}")

def validate(model, VALIDATION_DIRS, Transcript):
    # Preprocess the validation data
    validation_ft = "validation.ft"
    VALIDATION_DIRS = [os.path.join(DATA_DIR, filename) for filename in VALIDATION_DIRS]
    print(f"VALIDATION_DIRS: {VALIDATION_DIRS}")
    make_fasttext(VALIDATION_DIRS, validation_ft, Transcript, mode = "VAL")

    # Measure performance on validation set
    validation_performance = model.test('validation.ft')
    print(f"Performance on validation set "
          f"({validation_performance[0]} entries): "
          f"\t{validation_performance[1]*100:.2f}% accuracy")

def test(model, TEST_DIRS, Transcript):
    # Preprocess the test data
    test_ft = "test.ft"
    TEST_DIRS = [os.path.join(DATA_DIR, filename) for filename in TEST_DIRS]
    print(f"TEST_DIRS: {TEST_DIRS}")
    make_fasttext(TEST_DIRS, test_ft, Transcript, mode = "TEST")

    # Measure performance on test set
    test_performance = model.test('test.ft')
    print(f"Performance on test set "
          f"({test_performance[0]} entries): "
          f"\t{test_performance[1]*100:.2f}% accuracy")

def train():
    start = time.time()

    TRAIN_DIRS = []
    VALIDATION_DIRS = []
    TEST_DIRS = []
    for i in range(0, 13):
        filename = f"sw{i:02}utt"
        if i < 7:#< 7:
            TRAIN_DIRS.append(filename)
        elif i < 10:#< 10:
            VALIDATION_DIRS.append(filename)
        else:
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
    make_fasttext(TRAIN_DIRS, train_ft, Transcript, mode = "TRAIN")

    print(f"{TRAIN_MALE_ENTRIES} male partial conversations in training")
    print(f"{TRAIN_FEMALE_ENTRIES} female partial conversations in training")
    print(f"{TRAIN_DISCARDED_ENTRIES} discarded partial conversations in training")

    word_disparities = defaultdict(float)

    print("Total words: " + str(sum(wordcounts.values())))
    print("Unique words: " + str(len(wordcounts.keys())))

    words_filtered = 0

    for lemma, count in wordcounts.items():
        # 72 is about 0.01% of the corpus
        if count > 72:
            disparity = abs((male_wordcounts[lemma]/.545) 
                            - (female_wordcounts[lemma]/.455)) / count
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
                                      epoch=4,
                                      )
    train_performance = model.test('train.ft')
    print(f"Performance on train set "
          f"({train_performance[0]} entries): "
          f"\t{train_performance[1]*100:.2f}% accuracy")

    model.save_model("model.bin")

    trained = time.time()
    print(f"Finished training in {trained - start:.2f} seconds")

    validate(model, VALIDATION_DIRS, Transcript)

    print(f"{VAL_MALE_ENTRIES} male partial conversations in validation")
    print(f"{VAL_FEMALE_ENTRIES} female partial conversations in validation")
    print(f"{VAL_DISCARDED_ENTRIES} discarded partial conversations in validation")

    # No peeking
    # test(model, TEST_DIRS, Transcript)

    end = time.time()
    print(f"Finished overall in {end - start:.2f} seconds")

if __name__ == "__main__":
    train()
