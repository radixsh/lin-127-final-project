import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
import statistics
from functools import reduce

tokenizer = RegexpTokenizer(r'\w+')
syllable_tokenizer = SyllableTokenizer()

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

    features['length'] = len(sent)
    # Length of the sentence "bye-bye.", anything smaller is probably not useful.
    if features['length'] <=  8:
        return None
    
    # Count annotations
    features['num_laughs'] = sent.count('<laughter>')
    features['num_coughs'] = sent.count('<cough>')
    features['num_throat_clears'] = sent.count('<throat_clearing>')
    features['num_breathing'] = sent.count('<breathing>')
    features['num_coordconj'] = sent.count('{c')
    features['num_discourse_marks'] = sent.count('{d')
    features['num_filled_pauses'] = sent.count('{f')
    features['interrupted'] = " -" in sent
    
    # Remove the annotations, ignoring the rest, which don't provide enough info
    sent = purge_enclosed(sent)

    # Count [ ... + ... ], the speech errors + corrections, 
    # replace with the correction and remove extra whitespace
    # This version will get saved out with the sentence: tag.
    features['num_stutters'] = sent.count('+')
    features['sentence'] = " ".join(clean_square_brackets(sent).split())

    # Make lowercase, continue chopping up the sentence
    sent = features['sentence'].lower()

    # Count the number of commas before nuking punctuation
    features['num_commas'] = sent.count(',')

    # Remove all punctuation
    sent.translate(str.maketrans('', '', string.punctuation))

    features['mentions_children'] = reduce(lambda a,b: a or b, 
                               [word in sent for word in ["child",
                                                          "kid",
                                                          "son",
                                                          "daughter",
                                                          "baby",
                                                          ]])
    
    # Tokenize very aggressively, things like "you're" become ["you", "re"]
    tokens = tokenizer(sent)

    # Count words
    features['num_words'] = len(tokens)

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

    features['num_prons'] = num_prons
    features['fp_prons'] /= num_prons
    features['sp_prons'] /= num_prons
    features['male_tp_prons'] /= num_prons
    features['female_tp_prons'] /= num_prons
    features['neutral_tp_prons'] /= num_prons

    # Analyze word lengths slightly more than naively
    word_lens = map(len, tokens)
    features['word_lens'] = word_lens

    avg_word_length = statistics.mean(word_lens)
    median_word_length = statistics.median(word_lens)
    stddev_word_length = statistics.stdev(word_lens)

    nontrivial_word_lens = [n for n in word_lens if n > 3]
    num_nontrivial_words = len(nontrivial_word_lens)

    avg_nontrivial_word_length = statistics.mean(nontrivial_word_lens)
    median_nontrivial_word_length = statistics.median(nontrivial_word_lens)
    stddev_nontrivial_word_length = statistics.stdev(nontrivial_word_lens)

    longest_word_len = max(nontrivial_word_lens)
    # about ~16% of english words are 8 or longer characters
    # silly implementation
    long_word_cutoff = 8
    num_long_words = reduce(
        lambda a, b: a + 1 if b > long_word_cutoff else a, 
        nontrivial_word_lens
    )

    # Analyze syllables, turn each word into the number of syllables it contains
    # This has some small issues with apostrophes, since we split on them before. 
    syllable_counts = [len(SyllableTokenizer.tokenize(word)) for word in tokens]
    
    total_syllables = sum(syllable_counts)

    max_syllables = max(syllable_counts)
    avg_num_syllables = statistics.mean(syllable_counts)
    median_num_syllables = statistics.median(syllable_counts)
    stddev_num_syllables = statistics.stdev(syllable_counts)

    nontrivial_syllable_counts = [n for n in syllable_counts if n > 1]
    avg_nontrivial_syllables = statistics.mean(nontrivial_syllable_counts)
    median_nontrivial_syllables = statistics.median(nontrivial_syllable_counts)
    stddev_nontrivial_syllables = statistics.stdev(nontrivial_syllable_counts)

    num_big_words = reduce(
        lambda a, b: a + 1 if b > 3 else a, 
        nontrivial_word_lens
    )

    return sent + '\n'


def format_conv(conv):

    return conv + '\n'