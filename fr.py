import kivy
import cv2, os
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')
Config.set('graphics','show_cursor','1')

def FaceGenerator():
# open the camera and capture video
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    ID = 0
    sample_number = 0 # a counter that counts the number of pictures for each person in the database

    # detecting the face and draw rectangle on it
    while (True):
        retval,image = cam.read() # reading image from cam
        print (np.shape(image))
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # converting image to gray image
        faces = face_detector.detectMultiScale(gray_image,1.3,5)
        ''' detectMultiScale, detects objects of different sizes in the 
        input image.
        the detected objects are returned as a list of rectangles
        '''
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            sample_number=sample_number+1
        # saving the captured face in the facebase folder

            cv2.imwrite('Trainer/User.'+str(ID)+'.'+str(sample_number)+'.jpg',
            gray_image[y:y+h,x:x+w])
    # this loop drawing a rectabgle on the face while the cam is open 
        cv2.imshow('frame',image)
        k = cv2.waitKey(1);  0xff
        if k == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    output = "Succesfully created trainning set"
    return output

class ScreenOne(Screen):
    pass

class ScreenTwo(Screen):
    pass

class ScreenThree(Screen):
        pass


class ScreenManagement(ScreenManager):
    pass


sm = Builder.load_file("facerecognition.kv")

class FaceRecognitionApp(App):
    def build(self):
        FaceGenerator()
        return sm

if __name__=="__main__":
    FaceRecognitionApp().run()