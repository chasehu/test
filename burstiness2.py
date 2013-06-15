import csv as csv 
from operator import itemgetter
from nltk.stem.wordnet import WordNetLemmatizer
import re
import sys
import nltk.data, nltk.tag
from nltk.corpus import stopwords
import time

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
#import nltk.SvmClassifier


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scikits.learn import svm
from scikits.learn.svm import SVR
from scikits.learn.grid_search import GridSearchCV
from scikits.learn.cross_val import LeaveOneOut
from scikits.learn.metrics import confusion_matrix
from scikits.learn import cross_val

import cPickle


def load_sent_classifier():
	return cPickle.load(open('smilesadclassifier.pkl', 'rb'))

def load_master_burst():
	return cPickle.load(open('burst.pkl', 'rb'))



def removeSpecialTokens(cur_line):
	cur_tokens = cur_line.replace('\\n',' ').split(' ')
	rs_tokens = []

	for tk in cur_tokens:
		if tk == None or len(tk) == 0:
			continue

		if tk == 'RT' or tk[0] == '@' or tk[0] == '\\':
			continue
		if len(tk) > 3 and tk[1:3] == '\\x':
			continue
		if tk[0:4] == 'http':
			continue
		else:
			rs_tokens.append(tk)


	return ' '.join(rs_tokens)

def findTweetTopic(tweet): 
	lmtzr = WordNetLemmatizer()

	#tagger = nltk.data.load(nltk.tag._POS_TAGGER)
	tokenized = nltk.wordpunct_tokenize(tweet)

	# remove stopwords
	filtered_words = [w for w in tokenized if not w in stopwords.words('english') and len(w) > 2]

	# identify parts of speech, pull out nouns
	taggedTweet = nltk.pos_tag(filtered_words)
	nouns = []
	if taggedTweet != []: 
		for word in taggedTweet:
			pos = word[1]
			if(pos[0:2] == "NN" or pos == "FW"):
				word = str(word[0])
				word = lmtzr.lemmatize(word)
				nouns.append([word, 0, 0, 0.0])

	return nouns

# returns list of nouns in word
def findTweetTopic2(tweet):

	lmtzr = WordNetLemmatizer()

	#tagger = nltk.data.load(nltk.tag._POS_TAGGER)
	tokenized = nltk.wordpunct_tokenize(tweet)

	# remove stopwords
	filtered_words = [w for w in tokenized if not w in stopwords.words('english') and len(w) > 2]

	# identify parts of speech, pull out nouns
	taggedTweet = nltk.pos_tag(filtered_words)
	nouns = []
	if taggedTweet != []: 
		for word in taggedTweet:
			pos = word[1]
			if(pos[0:2] == "NN" or pos == "FW"):
				word = str(word[0])
				word = lmtzr.lemmatize(word)
				nouns.append(word)

	return nouns
'''
Processes the corpus into a readable format.
'''
def processCorpus(csvfile):
	csv_file_object = csv.reader(open('G:\\Users\\Russell\\Dropbox\\PubHack\\timelinecorpus.csv', 'r')) 

	myCorpus = []
	#rowCount = 0

	f_write = open('cleancorpus.txt','w')

	for row in csv_file_object:
		document = ''
		for tweet in row:
			if tweet == '':
				continue
			document += removeSpecialTokens(tweet.lstrip('\'\"[').split('\',')[0].split('\",')[0]) + ' '
		myCorpus.append(document)
		print>>f_write, document

	return myCorpus

def bursty(text):
	#process the corpus
	#myCorpus = processCorpus('timelinecorpus.csv')

	if text == '':		
		print "Please input text"
		text = sys.stdin.readline()

	tstart = time.clock()

	#stdin version
	nouns = findTweetTopic2(removeSpecialTokens(text))
	print time.clock() - tstart


	#load dictionary
	master_burst = load_master_burst()
	#print master_burst['chase']

	noun_burst = []

	print 'nouns: ' + nouns[0]
	for noun in nouns:
		burst = master_burst.get(noun, None)
		if burst != None:
			noun_burst.append([noun,burst[0],burst[1],burst[2]])


	noun_burst = sorted(noun_burst, key=itemgetter(3), reverse=True)


	tfin = time.clock() - tstart
	print "Time:" + str(tfin)

	#print noun_burst
	
	try:
		return noun_burst[0][0]
	except:
		return ''

if __name__ == "__main__":
	bursty('')

