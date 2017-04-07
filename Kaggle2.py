# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:01:35 2017

@author: DK
"""
import gc
import io
from zipfile import ZipFile
import csv
import urllib.request
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import nltk
import re
from string import punctuation
import sklearn.model_selection
train = pd.read_csv("D://train.csv")
#test=  pd.read_csv("D://test.csv")
train = train.fillna('empty')
#test = test.fillna('empty')
from sklearn.model_selection import train_test_split
#1.1 Разделить train.csv на тренировочный сет и тестовый сет. Соотношение 70/30
np.random.seed(1)
df_train, df_test = train_test_split(train, test_size = 0.3)
#1)Загрузка датасета word2vec.csv
Word2Vec=pd.read_csv("D://final//wordvectors//glove.twitter.27B.200d.txt",sep=" ",header=None)
#точно так же, как и загружали test.csv и train.csv - их надо как source_train и source_test сохранить
Columns=pd.read_csv("D://final//wordvectors//words.csv",sep=";",header=None,encoding="utf-8",nrows=300001)[0]#single out column names to the separate vector
Word2Vec=pd.read_csv("D://final//wordvectors//eigenwords.csv",sep=";",decimal=",",header=None,encoding="ascii",usecols=range(1,201))
Word2Vec=Word2Vec.values
#delete all columns with non-alphabetical and non-numeric characters


gc.collect()
#Anything EXCEPT FOR letters A-Z a-z should be deleted in order to should 
#2)для каждой пары предложений:
#2.1)почистить знаки препинания
#скопировать text_to_wordlist из "the importance of cleaning text", 
def text_to_wordlist(text):#making replacements
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"What's", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    text = ''.join([c for c in text if c not in punctuation])
    text=text.lower()#to lower case
    # Return a list of words
    return(text)
#скопировать process_questions. 
def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in questions:
        question_list.append(text_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(dataframe) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))
            
train_question1 = []
process_questions(train_question1, df_train.question1, 'train_question1', df_train)
train_question2 = []
process_questions(train_question2, df_train.question2, 'train_question2', df_train)
test_question1 = []
process_questions(test_question1, df_test.question1, 'test_question1', df_test)
test_question2 = []
process_questions(test_question2, df_test.question2, 'test_question2', df_test)
#train_question1, train_question2, test_question1 и test_question2 получать так, как там,
# только последние 2 - из нашего собственного test сета
#2.2)убрать в 1м предложении каждое из слов, которые есть во 2м предложении, одновременно те же слова убирать и во 2м предложении
def DeleteCommonWords(string1, string2):
    toremove = []
    Words1=str.split(string1)#новый объект: массив всех слов из первой строки. 
    Words2=str.split(string2)#То же и для второй
    for i in iter(Words1):
        if i in Words2:#i встречается в Words2 хотя бы на 1 позиции)
            toremove.append(i)
            Words2.remove(i)
    #All values in toremove should occur ONLY min(number of occurences in Words1,number of occurences in Words2) times
    
    for i in toremove:
        Words1.remove(i)
        
    return Words1,Words2


dictionary=dict()
i=0
for word in Columns:
    dictionary[word] = i
    i=i+1
#Each word vector is to de called as Word2Vec[dictionary(word),:]
def vector(word):#  MIND that if we ALREADY APPLIED function lower() to all inputs of this function, this function takes ONLY LOWERCASED WORDS as arguments
     BigNumber=9999999#very big number
     index=BigNumber#initialise index by big number
     Variants=[word.upper(),word.lower(),word[0].upper()+word.lower()]#all possible variants
     for candidate in Variants:
         if candidate in dictionary:
             index=min(index,dictionary[candidate])#looking for the most probable variant - variant with the lowest index
   #This is done in case that we somehow lowercased the word which is more often met in upper case,
   #or with only first letter in upper case.
   #As the words in dictionary are sorted by frequency, function chooses most probable variant
     if index==BigNumber:
         index=0
     word=Word2Vec[index,:]
     return word
def QuestionPairsToVectorMatrix(train_question1,train_question2):
    #мы удалили все лишние слова
#1е предложение представить как среднее арифметическое векторов всех слов, то же и для 2го предложения.
    difference=np.zeros((200,len(train_question1)))
# записать разность этих 2 векторов, деленную на 2(для нормировки) в матрицу с меткой 0 либо 1(дубликат или нет)
    for i in range(0,len(train_question1)):#for each training example
        if i%1000==0:
            print('index ' + str(i))
        train_question1_words, train_question2_words= DeleteCommonWords(train_question1[i],train_question2[i])
        word1 = np.repeat(0,200)
        word2=np.repeat(0,200)
        if len(train_question1_words)>0:
            for word in train_question1_words:
                word1=word1+vector(word)
            word1=word1/len(train_question1_words)
        if len(train_question2_words)>0:
            for word in train_question2_words:
                word2=word2+vector(word)
            word2=word2/len(train_question2_words)
        newvector=(word1-word2)/2
        difference[:,i]=newvector
    return difference
#we are going to use QuestionPairsToVectorMatrix with df_train.is_duplicate for labels
def loss(prediction,answer):
    return abs(prediction-answer)#

train_labels=df_train.is_duplicate
train_data=QuestionPairsToVectorMatrix(train_question1,train_question2)
test_labels=df_test.is_duplicate
test_data=QuestionPairsToVectorMatrix(test_question1,test_question2)
#2.6) тренировать нейросеть доя этих вот 100: 1 скрытый слой, активация сигмоидом и т пimport nltk
