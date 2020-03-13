import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re
class FileMngr:

    def __init__(self, newPath):
        self.path = newPath
        self.pandafile = pd.read_csv(self.path)

    def loadFile(self):
        self.file = open(self.path, "r")
        self.text = self.file.read()
        self.file.close()
        

    def readFile(self):
        for x in self.text:
            print(x)

    def checkFile(self):
        print("no of tweets -- no of columns")
        print(self.pandafile.shape)
        print("HEADERS          MISSING DATA")
        print(self.pandafile.isnull().sum())
        print("HEADERS          DATA TYPES")
        print(self.pandafile.dtypes)
        print("CLASS          NUMBER OF")
        print(self.pandafile['class'].value_counts())
        
    def chartFile(self):
        print(self.pandafile.head())

    def cleanFile(self):
        self.pandafile['tweet'] = self.pandafile['tweet'].apply(lambda x : self.translateTags(x))
        self.pandafile['tweet'] = self.pandafile['tweet'].apply(lambda x : self.removeHTML(x))
        self.pandafile['tweet'] = self.pandafile['tweet'].apply(lambda x : self.removePunct(x))
        tokenize = RegexpTokenizer(r'\w+')
        self.pandafile['tweet'] = self.pandafile['tweet'].apply(lambda x : tokenize.tokenize(x.lower()))
        self.pandafile['tweet'] = self.pandafile['tweet'].apply(lambda x : self.lemmatizeText(x))
        self.pandafile['tweet'] = self.pandafile['tweet'].apply(lambda x : self.removeNum(x))
        print(self.pandafile.head())

    def removeNum(self, text):
        cleanText = ''.join([i for i in text if not i.isdigit()])
        return cleanText

    def createFile(self):
        self.pandafile.to_csv(r'C:\Users\stake\Desktop\Detecting Hate Speech\Program\data\cleanedData.csv', index=False)

    def getText(self):
        return self.pandafile

    def lemmatizeText(self, text):
        lemmatizer = WordNetLemmatizer()
        cleanText = " ".join([lemmatizer.lemmatize(i) for i in text])
        return cleanText

    def translateTags(self, text):
        httpLinks = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mentionTags = '@[\w@\-]+'
        hashTags = '#[\w#;&,\-]+'
        cleanText = re.sub(mentionTags, 'MENTION', text)
        cleanText = re.sub(hashTags, 'HASHTAG', cleanText)
        cleanText = re.sub(httpLinks, 'HTTPLINK', cleanText)
        return cleanText

    def removeHTML(self, text):
        newText = BeautifulSoup(text, "lxml")
        cleanText = newText.get_text()
        return cleanText

    def removePunct(self, text):
        cleanText = "".join([c for c in text if c not in string.punctuation])
        return cleanText

