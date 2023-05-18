import json 
import random 
import pickle   
import numpy as np 
import nltk
from nltk.stem  import WordNetLemmatizer
from keras.models import load_model
import time

import customtkinter 


############## Xử lý model #################

intents = json.loads(open('data_train.json',errors='ignore').read())
lemmatizer = WordNetLemmatizer()

# load data
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# load model 
model = load_model('chatboxfitness.h5')

# chuyển từ về dạng nghĩa gốc
def clean_word(sentences):
    word_list = nltk.word_tokenize(sentences)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    return word_list


# hot encode 
def bag_of_word (sentences):
    bag = []
    sentences=clean_word(sentences)
    for word in words :
        bag.append(1) if word in sentences else bag.append(0)    
    return np.array(bag)


# dự đoán kết quả
def predict_result(sentences):
    bag = bag_of_word(sentences)
    results = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    # bỏ qua các kết quả có sai số thấp hơn 0.25
    result = [[index,res] for index , res in enumerate(results) if res > ERROR_THRESHOLD ]
    # sắp xếp theo thứ tự giảm dần 
    result.sort(key=lambda x: x[1] , reverse=True)
    
    # trả về các kết quả dự đoán được
    result_list = []
    for r in result :
        result_list.append({'intent':classes[r[0]] , 'probability' : r[1] })
    print(result_list)
    return result_list

# phản hồi 
def get_response(intent_list , intent_json):
    tag = intent_list[0]['intent'] # lấy giá trị có trọng số lớn nhất
    list_intent_json = intent_json['intents']
    for i in list_intent_json:
        if tag == i['tag']:
            result =random.choice(i['responses'])
            break
    # kiểm tra xác xuất : nếu <0.6 ==> không hiểu   
    pro = intent_list[0]['probability']
    if pro < 0.6 :
        result = "I'm sorry, I don't understand what you mean. Could you please clarify?"

    return result


############### GUI #####################

   

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("800x500")

# frame 
frame1 = customtkinter.CTkFrame(master = root )

frame2 = customtkinter.CTkFrame(master = root)

# function 
def select_frame(name):
    if name == "page1":
        frame1.pack(padx=10, pady=10,fill="both" ,expand = True)
    else :
        frame1.pack_forget()
    if name == "page2":
        frame2.pack(padx=10, pady=10,fill="both" ,expand = True)
    else :
        frame2.pack_forget()

def select_frame_1 ():
    select_frame("page1")

def select_frame_2 ():
    select_frame("page2")

def change_appearance_mode_event(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)

def Quit():
    root.destroy()

# side bar frame 

option_mod = customtkinter.CTkFrame(master = root)
option_mod.pack(padx=10, pady=10, fill="both" ,expand = True ,side = 'left')

label_1_fm = customtkinter.CTkLabel(master = option_mod , text="Option" , font = ("Roboto",24))
label_1_fm.grid(row=0, column=0, padx=20, pady=(20, 10))

button_1_fm = customtkinter.CTkButton(master = option_mod ,text="Quit",command=Quit)
button_1_fm.grid(row=1, column=0, padx=20, pady=10)

option_fm = customtkinter.CTkOptionMenu(master = option_mod , values=["Dark","Light","System"],command=change_appearance_mode_event)
option_fm.grid(row=5, column=0, padx=20, pady=10)

# build frame 1 
label_1_f1 = customtkinter.CTkLabel(master = frame1 , text="Information" , font = ("Roboto",24))
label_1_f1.pack(pady = 12 , padx = 10)

entry_1_f1 = customtkinter.CTkEntry(master = frame1 , placeholder_text="Name" )
entry_1_f1.pack(pady = 12 , padx = 10)

entry_2_f1 = customtkinter.CTkEntry(master = frame1 , placeholder_text="Age" )
entry_2_f1.pack(pady = 12 , padx = 10)

entry_3_f1 = customtkinter.CTkEntry(master = frame1 , placeholder_text="Height")
entry_3_f1.pack(pady = 12 , padx = 10)

entry_4_f1 = customtkinter.CTkEntry(master = frame1 , placeholder_text="Weight")
entry_4_f1.pack(pady = 12 , padx = 10)

entry_5_f1 = customtkinter.CTkEntry(master = frame1 , placeholder_text="Female / Male")
entry_5_f1.pack(pady = 12 , padx = 10)

button_1_f1 = customtkinter.CTkButton(master=frame1 , text="Start", font=("Roboto",24),command=select_frame_2)
button_1_f1.pack(pady= 12 , padx = 10)



# build frame 2

textbox_1_f2 = customtkinter.CTkTextbox(master = frame2 , width=500 , height=400 ,corner_radius=10 ,font=("Roboto",15),wrap = 'word')
textbox_1_f2.grid(row=0 ,column=0 ,padx =40 , pady = 20)

entry_1_f2 = customtkinter.CTkEntry(master = frame2 ,placeholder_text="Type your questions here !!" , width=400 ,height=33,font=("Roboto",15))
entry_1_f2.grid(row=1, column=0 ,padx =40 , pady = 0,sticky = 'w')

def chat_message(event = None): 

    mess = entry_1_f2.get()
    textbox_1_f2.insert(customtkinter.END,"You :  " + mess+ '\n' +'\n')
    entry_1_f2.delete(0 , len(mess))
    # get result from model
    result_list = predict_result(mess)
    responses = get_response(result_list, intents)
    textbox_1_f2.tag_config("tagName",foreground='green')
    textbox_1_f2.insert(customtkinter.END, "ProTeenTeen :  " + responses + '\n' +'\n',"tagName")
    


button_1_f2 = customtkinter.CTkButton(master = frame2 ,text="Send" ,font=("Roboto",20) , command=chat_message ,width=35,height=20)
button_1_f2.place(x =465 , y = 440)

# keyboard enter to send messages
root.bind('<Return>',chat_message)

select_frame_1()

root.mainloop()

