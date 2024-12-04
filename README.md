# LIN 127 Project

**Goal: determine gender and education level**

First, we will label every **sentence** with the gender of its speaker, and we
will also manually label features

### Labels:
Gender for now, education level later

### Features to guess gender and education level:
- sentence length (number of words in each verbalization)
- presence of fillers ("uh", "um")
- number of questions vs statements
- 1st person vs 3rd person pronouns ("I"/"me" vs "we" vs "you" vs "they")
- number of times they got interrupted???

See if the data is skewed/biased (are there more examples of men/men than
examples of women/women calls?)
- To mitigate this, use oversampling techniques: randomly oversample data from
  the less represented group

Then feed this to FastText and bingo bango bongo

Use built-in Transcript object in swda.py to combine
