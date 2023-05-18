import json 
import random 
import pickle   
import numpy as np 

import nltk
from nltk.stem  import WordNetLemmatizer

from keras.models import Sequential , Model
from keras.layers import Dense,Input,Reshape ,Flatten,Conv2D,MaxPooling2D,Normalization,LeakyReLU,Dropout,Activation , Embedding,Conv1D,MaxPooling1D
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

import matplotlib.pyplot as plt

#########################-----Data------################################

intents = json.loads(open('data_train.json',errors='ignore').read())

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
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letter]

# xoá các từ trùng nhau (thu nhỏ size của data cần được train) và sắp xếp lại 
words = sorted(set(words))

# xếp classes
classes = sorted(set(classes))

# lưu  data word và classes vào các file để sử dụng cho file main.py
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# hot encode pattern thành array có độ dài len(words) để làm input cho CNN
training = []
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
    training.append([bag,output_lable])

# sắp xếp lại tranining randomly 
random.shuffle(training)
# chuyển về numpy array vì tensorflow làm việc với numpy
training = np.array(training)

# tách train_data và train_lable
train_x = list(training[:,0]) # lấy tất cả giá trị của cột đầu tiên 
train_y = list(training[:,1]) # ___________________________thứ hai


##############------BUILD MODEL --------###########

model = Sequential()

# embededing
MAX_SEQUENCE_LENGTH = len(train_x[0]) 
VOCAB_SIZE = len(words) 
EMBED_SIZE = 20

model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH))

# bo CNN
# sử dụng bộ CONV1D vì đầu vào là mảng 1 chiều 
model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))    
#model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# bo ANN  

model.add(Flatten()) # tao ra mot vetor phang
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
model.summary()


#############-----COMPILE------##################


sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=["accuracy"])
history= model.fit(np.array(train_x),np.array(train_y),batch_size=5 ,epochs=200,verbose=1)

# save model
model.save('chatboxfitness.h5',history)


# hien thi do thi 
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()