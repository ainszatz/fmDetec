import cv2 as cv

face_cascade = cv.CascadeClassifier('data\\xml\\haarcascade_frontalface_default.xml')
mouth_cascade = cv.CascadeClassifier('data\\xml\\haarcascade_mcs_mouth.xml')

bw_threshold = 100
font = cv.FONT_HERSHEY_SIMPLEX
org = (30, 30)
default_color = (255, 255, 255) #white
weared_mask_font_color = (0, 255, 0) #green
not_weared_mask_font_color = (0, 0, 255) #red
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK"

cap = cv.VideoCapture(2)

while 1:
    ret, img = cap.read()
    img = cv.flip(img,1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    (thresh, black_and_white) = cv.threshold(gray, bw_threshold, 255, cv.THRESH_BINARY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)


    if(len(faces) == 0 and len(faces_bw) == 0):
        cv.putText(img, "No face found...", org, font, font_scale, default_color, thickness, cv.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        cv.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
        if(len(mouth_rects) == 0):
            cv.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if(y < my < y + h):
                    cv.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv.LINE_AA)
                    break

    cv.imshow('Mask Detection', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()