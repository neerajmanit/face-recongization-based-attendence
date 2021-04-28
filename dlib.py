import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import os
import dlib
import cv2
import matplotlib.pyplot as plt



model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load VGG Face model weights
model.load_weights('C:\\Users\\acer\\Downloads\\vgg_face_weights.h5')

model.summary()

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)


# fetching all the folder name
person_names=os.listdir('F:\Images_crop')

# creating the training data
x_train=[]
y_train=[]

# person dict used for mapping each person with their id
person_rep=dict()
for i,person in enumerate(person_names):
    folder=(os.listdir("F:\\Images_crop\\"+person))
    person_rep[i]=person
    for img_name in folder:
        final_path="F:\\Images_crop\\"+person+"\\"+img_name
        #print(final_path)
        img=load_img(final_path,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        print(img.shape)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        x_train.append(np.squeeze(K.eval(img_encode)).tolist())
        y_train.append(i)
        #img=cv2.imread(final_path)
        #cv2.imshow('frame',img)
        #cv2.waitKey(0)
        #cv2.destroyWindow('frame')
        #print(type(img))

# chech of mapped element
print(person_rep)

# convert into numpyarray
x_train=np.array(x_train)
y_train=np.array(y_train)


# creating test data checking model accuracy
x_test=[]
y_test=[]
#person_rep=dict()
for i,person in enumerate(person_names):
    folder=(os.listdir("F:\\Test_Images_crop\\"+person))
    #person_rep[i]=person
    for img_name in folder:
        final_path="F:\\Test_Images_crop\\"+person+"\\"+img_name
        #print(final_path)
        img=load_img(final_path,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        #print(img.shape)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        x_test.append(np.squeeze(K.eval(img_encode)).tolist())
        y_test.append(i)
        #img=cv2.imread(final_path)
        #cv2.imshow('frame',img)
        #cv2.waitKey(0)
        #cv2.destroyWindow('frame')
        print(type(img))


#convert into  numpyarray
x_test=np.array(x_test)
y_test=np.array(y_test)

# Save test and train data for later use
np.save('train_data',x_train)
np.save('train_labels',y_train)
np.save('test_data',x_test)
np.save('test_labels',y_test)

# Load saved data
x_train=np.load('train_data.npy')
y_train=np.load('train_labels.npy')
x_test=np.load('test_data.npy')
y_test=np.load('test_labels.npy')

# Softmax regressor to classify images based on encoding
classifier_model=Sequential()
classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.3))
classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=6,kernel_initializer='he_uniform'))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])

#fit the training data into model
# we can also check its accuracy on the basis of validation data
classifier_model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))


# Save model for later use
tf.keras.models.save_model(classifier_model,'F:/Face_Recognition/face_classifier_model.h5')


##################################################################################################################################################
##########################################Training Work Complete #############################################################



################################################### Now its time to reccognizer of model ###############################################
# 1 we collect picture with the help of Video camera  and extract face from  it it may be contain mutiple face
# 2 we use HCC for extraxting face
# 3 these faces use for extracting faces

# This Part is of Recognizer
# extract face images form video and save inside Test images
def extract_face_video():
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        'F:\Computer-Vision-with-Python\DATA\haarcascades\haarcascade_frontalface_default.xml')
    smapleNum=0
    while(True):
        ret,img=cam.read()
        faces = detector.detectMultiScale(img, 1.4, 6)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w + 10, y + h + 10), (255, 0, 0), 2)

            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite("F://faceimg//User." +  '.' + str(sampleNum) + ".jpg", img[y:y + h, x:x + w])

            cv2.imshow('frame', img)
        # wait for 1 miliseconds
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # break if the sample number is morethan 20
        elif sampleNum > 10:
            break
        fps = int(cam.get(5))
        print("fps:", fps)
    cam.release()
    cv2.destroyAllWindows()


def extract_face_img():
    pass


# Load  the model
classifier_model=tf.keras.models.load_model('F:/Face_Recognition/face_classifier_model.h5')


# Path to folder which contains images to be tested and predicted
path='F://'
test_images_path=path+'//Test_Images//'

#dnnFaceDetector=dlib.cnn_face_detection_model_v1("D://mmod_human_face_detector.dat")



def plot(img):
  plt.figure(figsize=(8,4))
  plt.imshow(img[:,:,::-1])
  plt.show()


#person_rep={0: 'Akshay Kumar',1: 'Nawazuddin Siddiqui',2: 'Salman Khan',3: 'Shahrukh Khan',4: 'Sunil Shetty',5:'Ganesh'}
#extract folder name of train images and map with their index

def extract_folder():
    pass


os.mkdir(path+'/Predictions')

# this is final folder after extract
path='F:\Test_Images'

list=os.listdir(path)
print(list)
for li in list:
    final_path=path+"\\"+li
    crop_img=load_img(final_path,target_size=(224,224))
    crop_img=img_to_array(crop_img)
    crop_img=np.expand_dims(crop_img,axis=0)
    crop_img=preprocess_input(crop_img)
    img_encode=vgg_face(crop_img)

    # Make Predictions
    embed=K.eval(img_encode)
    person=classifier_model.predict(embed)
    name=person_rep[np.argmax(person)]
    print(name)
    print(str(np.max(person)))
    img=cv2.imread(final_path)
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyWindow('frame')
    #os.remove(path+'/Test_Images/crop_img.jpg')
    #cv2.rectangle(img,(left,top),(right,bottom),(0,255,0), 2)
    #img=cv2.putText(img,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
    #img=cv2.putText(img,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)


#Due Work
#Handing of excel sheet
# Model
# Registration form
#developer Page