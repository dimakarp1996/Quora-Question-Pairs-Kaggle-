# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:01:35 2017

@author: DK
"""
from sklearn.neural_network import MLPClassifier
import gc
import numpy as np
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import re
from string import punctuation
import gensim
train = pd.read_csv("D://train.csv")
validation=  pd.read_csv("D://test.csv")#Validation are KAGGLE data for which we should pick examples.
train = train.fillna('empty')
validation = validation.fillna('empty')
from sklearn.model_selection import train_test_split
#1.1 Разделить train.csv на тренировочный сет и тестовый сет. Соотношение 70/30
np.random.seed(1)
df_train, df_test = train_test_split(train, test_size = 0.1)
#1)Загрузка датасета word2vec.csv
Word2Vec = gensim.models.KeyedVectors.load_word2vec_format('file://C:/Users/DK/GoogleNews-vectors-negative300.bin', binary=True)#Word2Vec embedding
embed_size=300#size of the embedding
#delete all columns with non-alphabetical and non-numeric characters
def vector(word):
        return Word2Vec[word] if word in Word2Vec else np.zeros(embed_size)

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
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
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
 #We pre-process train data, test data and validation data from Kaggle           
train_question1 = []
process_questions(train_question1, df_train.question1, 'train_question1', df_train)
train_question2 = []
process_questions(train_question2, df_train.question2, 'train_question2', df_train)
test_question1 = []
process_questions(test_question1, df_test.question1, 'test_question1', df_test)
test_question2 = []
process_questions(test_question2, df_test.question2, 'test_question2', df_test)
validation_question1=[]
process_questions(validation_question1, validation.question1, 'validation_question1', validation)
validation_question2=[]
process_questions(validation_question2, validation.question2, 'validation_question2', validation)
#train_question1, train_question2, test_question1 и test_question2 получать так, как там,
# только последние 2 - из нашего собственного test сета
#2.2)убрать в 1м предложении каждое из слов, которые есть во 2м предложении, одновременно те же слова убирать и во 2м предложении
#def DeleteCommonWords(string1, string2):
#    toremove = []
#    Words1=str.split(string1)#новый объект: массив всех слов из первой строки. 
#    Words2=str.split(string2)#То же и для второй
#    for i in iter(Words1):
#        if i in Words2:#i встречается в Words2 хотя бы на 1 позиции)
#            toremove.append(i)
#            Words2.remove(i)
#    #All values in toremove should occur ONLY min(number of occurences in Words1,number of occurences in Words2) times
#    
#    for i in toremove:
#        Words1.remove(i)
#        
#    return Words1,Words2,toremove
#from sklearn.ensemble import RandomForestClassifier
#def QuestionPairsToVectorMatrix(train_question1,train_question2):
#    #мы удалили все лишние слова
##1е предложение представить как среднее арифметическое векторов всех слов, то же и для 2го предложения.
#    difference=np.zeros((2*embed_size,len(train_question1)))
##записать один вектор поверх другого
#    for i in range(0,len(train_question1)):#for each training example
#        if i%1000==0:
#            print('index ' + str(i))
#        train_question1_words, train_question2_words,_= DeleteCommonWords(train_question1[i],train_question2[i])
#        word1 = np.repeat(0,embed_size)
#        word2=np.repeat(0,embed_size)
#        if len(train_question1_words)>0:
#            for word in train_question1_words:
#                word1=word1+vector(word)
#            word1=word1/len(train_question1_words)
#        if len(train_question2_words)>0:
#            for word in train_question2_words:
#                word2=word2+vector(word)
#            word2=word2/len(train_question2_words)
#        newvector=np.concatenate([word1,word2],axis=0)
#        difference[:,i]=newvector
#    return difference
##we are going to use QuestionPairsToVectorMatrix with df_train.is_duplicate for labels
#
#train_labels=df_train.is_duplicate
#train_data=QuestionPairsToVectorMatrix(train_question1,train_question2)
#test_labels=df_test.is_duplicate
#test_data=QuestionPairsToVectorMatrix(test_question1,test_question2)
#total_labels=train_labels+test_labels
#total_data=np.concatenate([train_data,test_data],axis=1)
##validation_data=QuestionPairsToVectorMatrix(validation_question1,validation_question2)
##while training on train data and checking on test data we achieved 0.73898274341 accuracy
#clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(300, 1), random_state=1,max_iter=200)
#A=clf.fit(list(train_data.T),list(train_labels))
#Prediction=A.predict(list(test_data.T))
#Accuracy=sum(Prediction==test_labels)/len(Prediction)
#
#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(train_data.T, train_labels)
#Accuracy_rf=rf.score(test_data.T,test_labels)
def avg_word(words):
    ans=np.zeros(embed_size)
    if len(words)>0:
        for i in range(len(words)):
            ans=ans+vector(words[i])
        ans=ans/(2*len(words))
    return ans
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
        
    return Words1,Words2,toremove
from sklearn.ensemble import RandomForestClassifier
def QuestionPairsToVectorMatrix(train_question1,train_question2,maxlen=99999999999999):
    #мы удалили все лишние слова
#1е предложение представить как среднее арифметическое векторов всех слов, то же и для 2го предложения.
    difference=np.zeros((3*embed_size,len(train_question1)))
#записать один вектор поверх другого
    for i in range(0,len(train_question1)):#for each training example
        if i%10000==0:
            print('index ' + str(i))
            #print(Accuracy_3)
        train_question1_words, train_question2_words,context_words= DeleteCommonWords(train_question1[i],train_question2[i])
        newvector=np.concatenate([avg_word(train_question1_words),
                                  avg_word(train_question2_words),
                                  avg_word(context_words)
                                            ],axis=0)
        difference[:,i]=newvector/4#to be from -1 and 1
    return difference
#we are going to use QuestionPairsToVectorMatrix with df_train.is_duplicate for labels

train_labels=df_train.is_duplicate
train_data=QuestionPairsToVectorMatrix(train_question1,train_question2)
test_labels=df_test.is_duplicate
test_data=QuestionPairsToVectorMatrix(test_question1,test_question2)
#while training on train data and checking on test data we achieved 0.73898274341 accuracy
print("using adam with 300 hidden layer size")#300 is also to be tried! #100 gave 0.803, 200 gave 0.817
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(300, 1), random_state=1,max_iter=200)
A=clf.fit(list(train_data.T),list(train_labels))
Prediction=A.predict(list(test_data.T))
Accuracy_3=sum(Prediction==test_labels)/len(Prediction)
print(Accuracy_3)
#FINAL PREDICTION
total_labels=np.concatenate([train_labels,test_labels])
total_data=np.concatenate([train_data,test_data],axis=1)
del(train_data,test_data)
A1=clf.fit(list(total_data.T),list(total_labels))
del(total_data)
gc.collect()
validation_data=QuestionPairsToVectorMatrix(validation_question1,validation_question2)
del(Word2Vec)
gc.collect()
print("began predicting")
Prediction1=A1.predict(list(validation_data.T[:500000,:]))
print("first 500000 predicted")
Prediction2=A1.predict(list(validation_data.T[500000:1000000,:]))
print("first 1000000 predicted")
Prediction3=A1.predict(list(validation_data.T[1000000:1500000,:]))
print("first 1500000 predicted")
Prediction4=A1.predict(list(validation_data.T[1500000:2000000,:]))
print("first 2000000 predicted")
Prediction5=A1.predict(list(validation_data.T[2000000:,:]))
print("all")
Prediction=np.concatenate([Prediction1,Prediction2,Prediction3,Prediction4,Prediction5],axis=0)
Answer=pd.DataFrame( {'test_id' : validation.test_id,'is_duplicate' : Prediction})
Answer=Answer[['test_id','is_duplicate']]
Answer.to_csv("D://answer.csv",index=False)
#IsRight=(test_labels==Prediction)
#Accuracy=sum(IsRight)/len(IsRight)



#timer
#import timeit
#tic=timeit.default_timer()
#toc=timeit.default_timer()
#toc - tic #elapsed time in seconds