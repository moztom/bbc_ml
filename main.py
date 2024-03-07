## Imports
import numpy as np
import nltk
import operator
import os
import sys
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk import ngrams

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

## CLI argument
if __name__ == "__main__":
    path = (sys.argv)

## Importing the data
# Fetching the path of each category
path_business= str(path[1])+'/bbc/business'
path_entertainment= str(path[1])+'/bbc/entertainment'
path_politics= str(path[1])+'/bbc/politics'
path_sport= str(path[1])+'/bbc/sport'
path_tech= str(path[1])+'/bbc/tech'

# Concatenate all text files from a directory into one dataset
def fill_dataset(path,dataset,category):
  for file in os.listdir(path):
    sample=open(path+"/"+file, errors='ignore').read().replace('\n',' ')
    dataset.append((sample,category))

dataset_full=[]

fill_dataset(path_business,dataset_full,'business')
fill_dataset(path_entertainment,dataset_full,'entertainment')
fill_dataset(path_politics,dataset_full,'politics')
fill_dataset(path_sport,dataset_full,'sport')
fill_dataset(path_tech,dataset_full,'tech')

## Split datasets 80:10:10 (train:test:dev)
def split(dataset):
  size_dataset=len(dataset)
  size_test=int(round(size_dataset*0.2,0))

  list_test_indices=random.sample(range(size_dataset), size_test)
  train_set=[]
  test_set=[]
  for i,example in enumerate(dataset):
    if i in list_test_indices: test_set.append(example)
    else: train_set.append(example)
  return train_set,test_set

train_set,test_set=split(dataset_full)

random.shuffle(train_set)

# Creating the Developement set
original_size_test=len(test_set)
size_dev=int(round(original_size_test*0.5,0))
list_dev_indices=random.sample(range(original_size_test), size_dev)
dev_set=[]
new_test_set=[]
for i,example in enumerate(test_set):
  if i in list_dev_indices: dev_set.append(example)
  else: new_test_set.append(example)
test_set=new_test_set
random.shuffle(dev_set)
random.shuffle(test_set)

## Preprocessing the dataset
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("''")
stopwords.add("-")
stopwords.add("(")
stopwords.add(")")
stopwords.add("'s")
stopwords.add("'m")
stopwords.add("n't")
stopwords.add("'ve")
stopwords.add(":")
stopwords.add("'d")

lemmatizer = nltk.stem.WordNetLemmatizer()

def get_list_tokens(string,ngram):
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      if ngram==True:
        if token in stopwords: continue
        else:
          list_tokens.append(lemmatizer.lemmatize(token).lower())
      else:
        list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

## Feature 1: Word Frequency
dict_word_frequency={}

def word_freq(dataset):
  for article in dataset:
    sentence_tokens=get_list_tokens(article[0],False)
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1

word_freq(train_set)

sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:2000]

vocabulary=[]
for word,frequency in sorted_list:
  vocabulary.append(word)

## Feature 2: N-Grams
def generate_ngrams(text, n):
    # Tokenize the text into words
    tokens = get_list_tokens(text,True)
    # Generate n-grams
    n_grams = ngrams(tokens, n)
    return [' '.join(grams) for grams in n_grams]

dict_ngram_frequency={}

def ngrams_freq(dataset):
    n = 2
    for article in dataset:
        # Extract n-grams
        n_grams = generate_ngrams(article[0], n)
        # Combine unigrams and n-grams
        for word in n_grams:
            #if word in stopwords: continue  # Assuming 'stopwords' is already defined
            if word not in dict_ngram_frequency: dict_ngram_frequency[word] = 1
            else: dict_ngram_frequency[word] += 1

ngrams_freq(train_set)

sorted_ngram_list = sorted(dict_ngram_frequency.items(), key=operator.itemgetter(1), reverse=True)[:2000]
vocabulary_ngram = [word for word, frequency in sorted_ngram_list]

## Feature 3: PoS Tagging
# Identify all unique PoS tags in the training set
all_pos_tags = []
for article in train_set:
    tokens = get_list_tokens(article[0],False)
    pos_tags = nltk.pos_tag(tokens)
    all_pos_tags.extend([tag for _, tag in pos_tags])
pos_tag_set = set(all_pos_tags)
pos_tag_index = {tag: idx for idx, tag in enumerate(pos_tag_set)}

# Function to get n-grams feature vector
def get_ngrams_features(vocabulary, string):
    vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary=vocabulary_ngram)
    vector = vectorizer.fit_transform([string]).toarray()
    return vector[0]

# Function to get PoS tag counts
def get_pos_features(string):
    tokens = get_list_tokens(string,False)
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = np.zeros(len(pos_tag_set))
    for _, tag in pos_tags:
        if tag in pos_tag_set:
            pos_counts[pos_tag_index[tag]] += 1
    return pos_counts

## Vectorising into train and test sets
def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string,False)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text

X_train=[]
Y_train=[]
for instance in train_set:
  word_freq_vector=get_vector_text(vocabulary,instance[0])
  ngrams_vector = get_ngrams_features(vocabulary_ngram, instance[0])
  pos_vector = get_pos_features(instance[0])
  combined_vector = np.concatenate((word_freq_vector, ngrams_vector, pos_vector))
  X_train.append(combined_vector)
  Y_train.append(instance[1])

## Feature Selection
fs=SelectKBest(chi2, k=500).fit(X_train, Y_train)
X_train_new = SelectKBest(chi2, k=500).fit_transform(X_train, Y_train)

## Training the classifier
def train_svm_classifier(training_set, vocabulary):
  svm_clf=MultinomialNB()
  svm_clf.fit(np.asarray(X_train_new),np.asarray(Y_train))
  return svm_clf

svm_clf=train_svm_classifier(train_set,vocabulary)

## Performance with test set
X_test=[]
Y_test=[]
for instance in test_set:
  word_freq_vector = get_vector_text(vocabulary, instance[0])
  ngrams_vector = get_ngrams_features(vocabulary_ngram, instance[0])
  pos_vector = get_pos_features(instance[0])
  combined_vector = np.concatenate((word_freq_vector, ngrams_vector, pos_vector))
  X_test.append(combined_vector)
  Y_test.append(instance[1])
X_test=np.asarray(X_test)
Y_test_gold=np.asarray(Y_test)

Y_predictions=svm_clf.predict(fs.transform(X_test))

accuracy=accuracy_score(Y_test_gold, Y_predictions)
print ("Accuracy: "+str(round(accuracy,3)))
print(classification_report(Y_test_gold, Y_predictions))