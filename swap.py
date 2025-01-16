import cv2
import numpy as np

def face_swap(image1, image2, output_image):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(r'C:\\Users\\iT-Market\\OneDrive\\Desktop\\co_tam\\haarcascade_frontalface_default.xml')

    faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces1) == 0 or len(faces2) == 0:
        print("Лица не найдены на одном из изображений.")  
    else:
        for (x1, y1, w1, h1) in faces1:
            face1 = img1[y1:y1+h1, x1:x1+w1]

            for (x2, y2, w2, h2) in faces2:
                face2 = img2[y2:y2+h2, x2:x2+w2]

                face2_resized = cv2.resize(face1, (w2, h2))

                img2[y2:y2+h2, x2:x2+w2] = face2_resized

        cv2.imwrite(output_image, img2)

        cv2.imshow('Result', img2)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

face_swap( 'img37.png','img36.png', 'result.png')
