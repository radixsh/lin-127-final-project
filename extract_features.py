import string
from collections import defaultdict
import statistics
from functools import reduce

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag



regex_tokenizer = RegexpTokenizer(r'\w+')
syllable_tokenizer = SyllableTokenizer()
lemmatizer = WordNetLemmatizer()

DIVISIVE_WORDS = ['husband', 'wonderful', 'wear', 'dress', 'wife', 'huhuh', 'goodness',
                  'dinner', 'tax', 'cook', 'woman', 'gosh', 'color', 'girl', 'god', 
                  'neat', 'mother', 'oh', 'vacation', 'flower', 'told', 'men', 'fish', 
                  'ours', 'university', 'love',]

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


def clean_square_brackets(text):
    stack = []
    result = ""
    i = 0

    while i < len(text):
        if text[i] == "[":
            # Start of a new bracketed phrase, push current result to stack
            stack.append(result)
            result = ""
            i += 1
        elif text[i] == "]":
            # End of a bracketed phrase, pop from stack
            if stack:
                before_bracket = stack.pop()
                # Split the current result by '+', and take the part after '+'
                parts = result.split("+")
                if len(parts) > 1:
                    result = before_bracket + parts[1].strip()
                else:
                    result = before_bracket + result
            i += 1
        else:
            # Accumulate characters in the current result
            result += text[i]
            i += 1

    return result.strip()

def purge_enclosed(text):
    stack = []
    result = []
    braces = {"(": ")", "{": "}", "<": ">"}
    open_braces = set(braces.keys())
    close_braces = set(braces.values())

    for char in text:
        if char in open_braces:
            # Start of a new enclosure, push to stack
            stack.append(char)
        elif char in close_braces:
            # End of an enclosure, pop from stack
            if stack and braces[stack[-1]] == char:
                stack.pop()
        elif not stack:
            # Only add to result if we're not inside any braces
            result.append(char)

    return ''.join(result).strip()

def format_sentence(sent):

    # Dict to store the features
    features = {}

    # Length of the sentence "bye-bye.", anything smaller is probably not useful.
    if len(sent) <=  8:
        return None
    
    # Count annotations
    features['laughs'] = '<laughter>' in sent
    features['coughs'] = '<cough>' in sent
    features['throat_clearing'] = '<throat_clearing>' in sent
    features['breathing'] = '<breathing>' in sent
    features['num_coordconj'] = sent.count('{c')
    features['num_discourse_marks'] = sent.count('{d')
    features['filled_pause'] = '{f' in sent
    features['multi_filled_pause'] = (sent.count('{d') >= 2)
    features['interrupted'] = " -" in sent
    
    # Remove the annotations, ignoring the rest, which don't provide enough info
    sent = purge_enclosed(sent)

    # Count [ ... + ... ], the speech errors + corrections, 
    # replace with the correction and remove extra whitespace
    # This version will get saved out with the sentence: tag.
    features['num_stutters'] = sent.count('+')
    clean_sent = " ".join(clean_square_brackets(sent).split())

    # Make lowercase, continue chopping up the sentence
    sent = clean_sent.lower()

    # Count the number of commas before nuking punctuation
    features['num_commas'] = sent.count(',')

    # Remove all punctuation
    sent = sent.translate(str.maketrans('', '', string.punctuation))

    features['mentions_children'] = reduce(lambda a,b: a or b, 
                               [word in sent for word in ["child",
                                                          "kid",
                                                          "son",
                                                          "daughter",
                                                          "baby",
                                                          ]])
    
    # Tokenize!
    tokens = nltk.word_tokenize(sent)

    lemmae = []

    for word in tokens:
        pos = get_wordnet_pos(word)
        lemmae.append(lemmatizer.lemmatize(word, pos=pos))

    for word in DIVISIVE_WORDS:
        if word in lemmae:
            features[f'has_{word}'] = "True"
        else:
            features[f'has_{word}'] = "False"

    # Categorize sentence length
    if len(tokens) < 7:
        features['sent_length'] = 'short'
    elif len(tokens) < 14:
        features['sent_length'] = 'medium'
    else:
        features['sent_length'] = 'long'

    # Count pronouns, loop is faster than .count here
    num_prons, fp_prons, sp_prons = 0,0,0 
    male_tp_prons, female_tp_prons, neutral_tp_prons = 0,0,0
    for word in tokens:
        if word == 'i':
            fp_prons += 1
            num_prons += 1
        elif word == 'you':
            sp_prons += 1
            num_prons += 1
        elif word in ['he', 'him', 'his']:
            male_tp_prons += 1
            num_prons += 1
        elif word in ['she', 'her', 'hers']:
            female_tp_prons += 1
            num_prons += 1
        elif word in ['they', 'them', 'theirs']:
            neutral_tp_prons += 1
            num_prons += 1

    features['primary_person_prons'] = "None"
    features['primary_gender_prons'] = "None"
    features['primary_fp_vs_sp_prons'] = "None"

    tp_prons = male_tp_prons + female_tp_prons + neutral_tp_prons

    if fp_prons > sp_prons:
        features['primary_fp_vs_sp_prons'] = "fp"
        if fp_prons > tp_prons:
            features['primary_person_prons'] = "fp"
        elif tp_prons > fp_prons:
            features['primary_person_prons'] = "tp"
    elif sp_prons > fp_prons:
        features['primary_fp_vs_sp_prons'] = "sp"
        if sp_prons > tp_prons:
            features['primary_person_prons'] = "sp"
        elif tp_prons > sp_prons:
            features['primary_person_prons'] = "tp"

    if neutral_tp_prons > male_tp_prons:
        if neutral_tp_prons > female_tp_prons:
            features['primary_gender_prons'] = "neutral"
        elif female_tp_prons > neutral_tp_prons:
            features['primary_gender_prons'] = "female"
    elif male_tp_prons > neutral_tp_prons:
        if male_tp_prons > female_tp_prons:
            features['primary_gender_prons'] = "male"
        elif female_tp_prons > male_tp_prons:
            features['primary_gender_prons'] = "female"


    # Analyze word lengths slightly more than naively
    word_lens = list(map(len, tokens))

    # This should account for skewed data where someone uses lots of small words
    if len(word_lens) > 0:
        avg_word_length = statistics.mean(word_lens)
    else:
        return None

    if avg_word_length >= 6:
        features['avg_word_length'] = "long"
    elif avg_word_length >= 3:
        features['avg_word_length'] = "medium"
    else:
        features['avg_word_length'] = "short"

    nontrivial_word_lens = [n for n in word_lens if n > 3]

    # This should be more representative of medium-length words
    if len(nontrivial_word_lens) > 0:
        median_nontrivial_word_length = statistics.median(nontrivial_word_lens)
    else:
        median_nontrivial_word_length = 0

    if median_nontrivial_word_length >= 7:
        features['median_nontrivial_word_length'] = "long"
    elif median_nontrivial_word_length >= 5:
        features['median_nontrivial_word_length'] = "medium"
    else:
        features['median_nontrivial_word_length'] = "short"


    # about ~16% of english words are 8 or longer characters
    # silly implementation
    features['has_long_word_by_chars'] = "False"
    for n in nontrivial_word_lens:
        if n >= 8:
            features['has_long_word_by_chars'] = "True"
            break

    # Analyze syllables, turn each word into the number of syllables it contains
    # This has some small issues with apostrophes, since we split on them before. 
    syllable_counts = [len(syllable_tokenizer.tokenize(word)) for word in tokens]
    
    total_syllables = sum(syllable_counts)

    if total_syllables > 30:
        features['total_syllables'] = "many"
    elif total_syllables > 15:
        features['total_syllables'] = "medium"
    else:
        features['total_syllables'] = "few"

    if len(syllable_counts) > 0:
        avg_num_syllables = statistics.mean(syllable_counts)
    else:
        return None

    if avg_num_syllables > 2:
        features['avg_num_syllables'] = "many"
    elif avg_num_syllables >= 1.5:
        features['avg_num_syllables'] = "medium"
    else:
        features['avg_num_syllables'] = "few"

    nontrivial_syllable_counts = [n for n in syllable_counts if n > 1]

    if len(nontrivial_syllable_counts) > 0:
        median_nontrivial_syllables = statistics.median(nontrivial_syllable_counts)
    else:
        median_nontrivial_syllables = 0

    if median_nontrivial_syllables > 6:
        features['median_nontrivial_syllables'] = "many"
    elif median_nontrivial_syllables >= 3:
        features['median_nontrivial_syllables'] = "medium"
    else:
        features['median_nontrivial_syllables'] = "few"

    features['has_trisyllabic_or_longer'] = "False"
    features['has_really_long'] = "False"

    for n in syllable_counts:
        if n >= 4:
            features['has_trisyllabic_or_longer'] = "True"
            features['has_really_long'] = "True"
            break
        elif n >= 3:
            features['has_trisyllabic_or_longer'] = "True"

    # Need to format features into a string.

    out = ""
    for key, value in features.items():
        out += str(key) + ":" + str(value) + " "

    out += "sentence: " + str(clean_sent) + "\n"

    return out


def format_conv(conv):

    return conv + '\n'