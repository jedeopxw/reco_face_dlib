import face_recognition
from flask import Flask, jsonify, request, redirect, json, render_template
import unidecode
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import os

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.df = pd.read_pickle('face_reco_database.pkl')
        self.df.astype('str')
        self.known_face_encodings = list(self.df.Encodings.values)

    
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True 
        
    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        ret, frame = self.video.read()
        #fps = self.video.get(cv2.CAP_PROP_FPS)
        #fps = "fps" + str(fps)
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

   
        rgb_small_frame = small_frame[:, :, ::-1]

    
        if self.process_this_frame:
        
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
            
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Non reconnu"

            
                if True in matches:
                    first_match_index = matches.index(True)
                    #name = self.known_face_names[first_match_index]
                    names = list(self.df.Prénom.values)
                    pnames = list(self.df.Nom.values)
                    name = names[first_match_index]+" "+pnames[first_match_index]
                    
                    


                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame


    
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
        
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

        
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #cv2.putText(frame, fps, (left + 6, bottom + 20), font, 1.0, (255, 255, 255), 1)
            #cv2.putText(frame, orga, (0,0), font, 1.0, (0, 0, 0), 1)
        
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


