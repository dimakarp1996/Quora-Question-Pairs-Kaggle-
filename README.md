# PROJECT NAME
Quora question pairs from Kaggle
# TASK
Build neural network, which identifies question duplicates from Quora questions
# DATASET
Training set has 6 columns: pair id, id of the first question, id of the second question, first question, second question, boolean variable "isDuplicate". Testing set has only pair id, first question and second question. Column "isDuplicate" needs to be predicted in the test set.

https://yadi.sk/d/v8ras2A93GcJzV link to the train set


https://yadi.sk/d/KkgteUcs3GcK47 link to the test set

# METRIC
I am going to use GLOVE word embeddings(or maybe some other embeddings) to construct word vectors
# BASELINE DESCRIPTION
For each question pair: 1) delete all duplicate words 2) calculate average from remaining words in the first question and in the second question 3)subtract one from another 4) train simple neural network with 1 hidden layer in order to provide nesessary results. 
Mind that question pairs should be artificially padded: for each question we should create many new, previously unseen question pairs? in order to get more non-duplicate pairs.


https://www.kaggle.com/c/quora-question-pairs
