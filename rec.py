import face_recognition
from flask import Flask, jsonify, request, redirect, json, render_template
import unidecode
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def face_detect(frame):
    # Resize to 1/4 for faster detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Find all the faces and face locations
    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    face_locations_ = []
    nfaces = len(face_locations)
    for top, right, bottom, left in face_locations:
        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        face_locations_.append((top, right, bottom, left))
        # Extract the region of the image that contains the face
        #face_image = frame[top:bottom, left:right]
        # Put a blurred face into the frame image
        #face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        #frame[top:bottom, left:right] = face_image
    
    return frame, face_locations_

def landmarks_detect(frame,face_locations):
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(frame,face_locations)

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(frame*0)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:
        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=3)

    return np.array(pil_image)


def encodings(frame,face_locations=None,draw=True):
    if not face_locations:
        face_locations = face_recognition.face_locations(frame, model="cnn")
    # facial encodings for faces in the image
    face_landmarks_list = face_recognition.face_landmarks(frame,face_locations)
    face_encodings_list = face_recognition.face_encodings(frame, face_locations)

    if draw:
        # Create a PIL imagedraw object
        pil_image = Image.fromarray(frame*0)
        d = ImageDraw.Draw(pil_image)
        nface = 0
        for face_landmarks,face_encoding in zip(face_landmarks_list,face_encodings_list):
            nface += 1
            i = 0
            # Let's trace facial feature in the image with encodings numbers
            for facial_feature in face_landmarks.keys():    
                for lm_point in face_landmarks[facial_feature]:
                    for y in range(3):
                        point = (lm_point[0]-5 ,lm_point[1]+ 10*y-20)
                        d.text(point,
                               '{}'.format(str(face_encoding[i%128])[5*y:5*(y+1)]), 
                               fill=(255, 255, 255, 255))
                        i +=1
        pil_image.show()
        
    return face_encodings_list,face_locations

def create_record(img):
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        pil_image = Image.fromarray(img)  
        d = ImageDraw.Draw(pil_image)
        d.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        pil_image.show()
        cols = ['Encodings','Nom','Prénom','Organisation','Description','Phrase Fun']
        vals = [[face_encoding] + ['NA']*5]
        for i, col in enumerate(cols[1:]):
            vals[0][i+1] = input('{}: '.format(col))

        new_record = pd.DataFrame(columns=cols,data=vals)
        return new_record

def add_record(new_record):
    try: 
        #Load existing if any
        df = pd.read_pickle('face_reco_database.pkl')
        df = df.append(new_record,ignore_index=True)
    except:
        df = new_record.copy()
    df.reset_index(inplace=True)
    df.set_index('index',inplace=True)
    df.to_pickle('face_reco_database.pkl')
    #print('Encodages enregistrés : {}'.format(df))

def webcam_record():
    video_capture = cv2.VideoCapture(0)
    while True:
        # Grab a single frame
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to stop video 
        if cv2.waitKey(1) == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    #cv2.imwrite('webcam_shot.jpg',frame)np.array(pil_image)
    new_record = create_record(frame)
    add_record(new_record)
    return 'Ok'

#webcam_record()
filename = 'last_image.jpg'
img = cv2.imread(filename)
new_record = create_record(img)
add_record(new_record)