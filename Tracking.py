import numpy as np
import cv2
import time

print("[INFO] loading Cascades...")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


print("[INFO] starting front cam...")
cap = cv2.VideoCapture(0)

print("[INFO] detectign faces...")

names = []

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left drawing
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right drawing
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left drawing
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right drawing
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = time.time() 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    localtime = time.asctime( time.localtime(time.time()) )
    # dt1=t2-t1
    print(localtime)
    #print('face detection time: ' + str(round(dt1, 3)) + ' secs') 

    for (x,y,w,h) in faces:
        draw_border(img, (x, y), (x+w, y+h), (51, 51, 255), 2, 12, 12)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        

        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
        smile = smile_cascade.detectMultiScale(roi_gray,5,8)
        for (ex,ey,ew,eh) in eyes:
            draw_border(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0), 2, 5, 5)
        for (sx,sy,sw,sh) in smile:
            draw_border(roi_color,(sx,sy),(sx+sw,sy+sh), (225,0,0), 2, 5, 5)

    cv2.imshow('img',img)
    k = cv2.waitKey(1);  0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()