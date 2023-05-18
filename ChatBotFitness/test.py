import json 
import random 
import pickle   
import numpy as np 

import nltk
from nltk.stem  import WordNetLemmatizer

from keras.models import Sequential , Model , load_model
from keras.layers import Dense,Input,Reshape ,Flatten,Conv2D,MaxPooling2D,Normalization,LeakyReLU,Dropout,Activation , Embedding
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

import matplotlib.pyplot as plt


intents = json.loads(open('data_test.json',errors='ignore').read())
words = [] # list các word có trong all pattern 
classes = [] # list các tag
documents = [] # list các words thuộc vào tag nào tương ứng
ignore_letter = ['?','.','!',','] # các ký tự cần loại bỏ

# thêm các từ theo từng tag vào document
for intent in intents['intents']:
    for patterns in intent['patterns']:
        word_list = nltk.word_tokenize(patterns) # tách từng từ trong patterns
        words.extend(word_list)
        documents.append((word_list, intent['tag'])) # để biết từ nào thuộc tag nào trong intent
        if intent['tag'] not in classes :
            classes.append(intent['tag']) # thêm tag vào doc


# đưa các từ về dạng nguyên thể để train cho bot (chỉ quan tâm đến nghĩa ,không quan tâm đế cú pháp)
lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# hot encode pattern thành array có độ dài len(words) để làm input cho CNN
test = []
output_empty = [0] * len(classes) # list co gia tri =0

for document in documents :
    bag = [] # hot encode cua pattern là array có độ dài len(words)
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in words :
        bag.append(1) if word in word_pattern else bag.append(0) # nếu có từ nào trong words trùng với trong pattern thì :1 else 0

    # hot encode tag  thành 1 0 để thể hiện vị trí của tag trong classes
    output_lable = list(output_empty) # copy
    output_lable[classes.index(document[1])] = 1 # output 1 0 ( 1 : thể hiện vị trí của tag)
    # thêm vào traininng làm input 
    test.append([bag,output_lable])

# sắp xếp lại tranining randomly 
random.shuffle(test)
# chuyển về numpy array vì tensorflow làm việc với numpy
test = np.array(test)

# tách train_data và train_lable
test_x = list(test[:,0]) # lấy tất cả giá trị của cột đầu tiên 
test_y = list(test[:,1]) # ___________________________thứ hai

# load model 
model = load_model('chatboxfitness.h5')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(np.array(test_x), np.array(test_y), batch_size=5)
print("test loss, test acc:", results)