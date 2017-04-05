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
def isEngLetterOrNumber(text):#check whether there are only uppercase letters, lowercase letters and numbers
    c=list("abcdefghijklmnopqrstuvwxyz '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in iter(text):
        if i not in c: return False
    return True

#1)Загрузка датасета word2vec.csv
Word2Vec=pd.read_csv("D://final//wordvectors//glove.twitter.27B.200d.txt",sep=" ",header=None)
#точно так же, как и загружали test.csv и train.csv - их надо как source_train и source_test сохранить
ColNames=Word2Vec.iloc[:,0]#single out column names to the separate vector
Word2Vec=Word2Vec.iloc[:,1:201]#remove it
#delete all columns with non-alphabetical and non-numeric characters
ToKeep=ColNames.apply(isEngLetterOrNumber)#all column names which consist only from eng.letter/number are to be deleted; maybe we should also preserve 1 token for unknown words?
Word2Vec=Word2Vec.values
Word2Vec=Word2Vec[ToKeep,:]
ColNames=ColNames[ToKeep]#only the selected words and word vectors are to remain
gc.collect()
#Anything EXCEPT FOR letters A-Z a-z should be deleted in order to should 
#2)для каждой пары предложений:
#2.1)почистить знаки препинания
#скопировать text_to_wordlist из "the importance of cleaning text", 
def text_to_wordlist(text):#making replacements
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
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
        if any(i in word for word in Words2):#i встречается в Words2 хотя бы на 1 позиции)
            toremove.append(i)
    for i in toremove:
        Words1.remove(i)
        Words2.remove(i)
    return Words1,Words2


TrainingVectors=NULL
dictionary=dict()
i=0
for word in ColNames1:
    dictionary[word] = i
    i=i+1
#Each word vector is to de called as Word2Vec[dictionary(word),:]
def vector(word):
     if word not in dictionary:
        word=Word2Vec[dictionary(unknown),:]
     else:
        word=Word2Vec[dictionary(word),:]
     return word
def QuestionPairsToVectorMatrix(train_question1,train_question2):
for i in range(0,train_question1.shape[0]):#for each training example
    train_question1_words, train_question2_words= DeleteCommonWords(train_question1[i],train_question2[i])
    word1 = rep(0,200)
    word2=rep(0,200)
    for word in train_question1_words:
        word1=word1+Word2Vec[dictionary(word),:]
    word1=word1/len(train_question1_words)
    for word in train_question2_words:
        word2=word2+Word2Vec[dictionary(word),:]
    word2=word2/len(train_question2_words)
    newvector=(word1-word2)/2
    if i=0:
        difference=newvector
    if i>0:
        difference=[difference,newvector]
    return difference
#мы удалили все лишние слова
#1е предложение представить как среднее арифметическое векторов всех слов, то же и для 2го предложения.
   
# записать разность этих 2 векторов, деленную на 2(для нормировки) в матрицу с меткой 0 либо 1(дубликат или нет)

TestingV
for i in range(0,test_question1.shape[0]):#for each training example
{
test_question1_words, test_question2_words= DeleteCommonWords(test_question1[i],test_question2[i])
}
2.5) ввести лосс-функцию как abs(y-y*), где y* - метка матрицы, y - предсказание нейросети.
2.6) тренировать нейросеть доя этих вот 100: 1 скрытый слой, активация сигмоидом и т пimport nltk
