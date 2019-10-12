from nltk.tokenize import word_tokenize, sent_tokenize as st #tokenizing sentence
from nltk.corpus import stopwords, wordnet #filtering sentence
from nltk.stem import PorterStemmer as ps, WordNetLemmatizer as wnl #stem filter to basic
from nltk.probability import FreqDist #show freq of words
from nltk.chunk import ne_chunk #drawing tree
from nltk.tag import pos_tag #detail tag
from nltk.classify import NaiveBayesClassifier, accuracy #trainimg and testing data

#import data from nltk data
from nltk.corpus import movie_reviews
import pickle

neg_review = []
pos_review = []

stop_words = stopwords.words('english')

for x in movie_reviews.categories():
    y = 0
    for i in movie_reviews.fileids(x):
        if x == 'neg':
            neg_review += [word for word in movie_reviews.words(i) if word not in stop_words]
        else:
            pos_review += [word for word in movie_reviews.words(i) if word not in stop_words]

        if y == 10: break
        else: y += 1

neg_review = neg_review[:5000]
pos_review = pos_review[:5000]

# FILTERING USING STEM AND LEMMATIZE
def extract(word, list_of_words, category):
    if word not in list_of_words:
        return ({word: False}, category)
    else:
        return ({word: True}, category)

stemmer = ps()
lemmatizer = wnl()

neg_review = [stemmer.stem(w) for w in neg_review]
pos_review = [stemmer.stem(w) for w in pos_review]

# [ key (for key in [arr] (condition if/else) ]

neg_review = [lemmatizer.lemmatize(w) for w in neg_review]
pos_review = [lemmatizer.lemmatize(w) for w in pos_review]

# EXTRACTING FEAUTURE (IMPORTANT)

neg_review = [extract(w, pos_review, 'negative') for w in neg_review]
pos_review = [extract(w, neg_review, 'positive') for w in pos_review]

# CREATE TRAINING AND TEST DATA
idx = int(.8 * 5000)
train_data = neg_review[:idx] + pos_review[:idx]
test_data = neg_review[idx:] + pos_review[idx:]

# TRAINING DATA TO CREATE A MODEL
model = NaiveBayesClassifier.train(train_data)

#CHECK ACCURACY
acc = accuracy(model, test_data)
print(acc * 100)

model.show_most_informative_features()

words_input = input('input words : ')

pos = 0
words = word_tokenize(words_input)

def extract_input(w):
    if w not in neg_review:
        return {w: True}
    return {w: False}

for w in words:
    w = stemmer.stem(w)
    w = lemmatizer.lemmatize(w)
    res = model.classify(extract_input(w))
    if res == 'positive': pos += 1

if pos > len(words) / 2: print('Review is Positive!')
else: print('Review is Negative!')

w = words[0]
for syn in wordnet.synsets(w):
    for s in syn.lemmas():
        print(s.name())
        for a in s.antonyms():
            print('\t' + a.name())

# SAVE DATA USING PICKLE
save_data = open('data.pickle', 'wb') # wb = write byte
pickle.dump(model, save_data)
save_data.close()

# LOAD DATA USING PICKLE
load_data = open('data.pickle', 'rb')
model = pickle.load(load_data)
load_data.close()