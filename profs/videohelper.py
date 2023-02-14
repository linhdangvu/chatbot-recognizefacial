import face_recognition
from datetime import datetime
import pickle
import cv2
import os

# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
lastPict= datetime.now();

#to find path of xml file containing haarCascade file
cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
fc = cv2.CascadeClassifier(cfp)

def searchFaces( frame ):
    global lastPict
    #convert image to Greyscale for HaarCascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray,
                                scaleFactor=1.1,
                                minNeighbors=6,
                                minSize=(60, 60),
                                flags=cv2.CASCADE_SCALE_IMAGE)
    encodings = face_recognition.face_encodings(frame)
    names = []
    if len(faces) > 0 :
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            #set name =unknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                count = {}
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    #increase count for the name we got
                    count[name] = count.get(name, 0) + 1
                    #set name which has highest count
                    name = max(count, key=count.get)
                    # will update the list of names
            names.append(name)
        
        
        now = datetime.now()
        diff= now - lastPict
        if len(names)>0 and diff.total_seconds()> 10:
            lastPict= now
            cv2.imwrite("temp/"+names[0]+"_"+now.strftime("%Y%m%d_%H%M%S")+".jpg", frame)
        
        # do loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            color= (0, 255, 0) if name!="Unknown" else (0, 0 ,255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
