import cv2
import os
from  pathlib import Path, PurePosixPath

#ignore hidden files (this which starts with '.')
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

# detects faces in the image, crops it around the face in (w x h) size square and resizes it
# returns 1 if face was detected, and 0 if not
def crop_detected_faces(file_path, cropped_file, faceCascade, notdetected):
    print(file_path)
    img = cv2.imread(str(file_path))
    gray = cv2.imread(str(file_path), 0)
    #recognized faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # crop the image around the face in (w x h) size square and resize it
    for (x, y, w, h) in faces:
        print(faces)
        cropped_img = cv2.resize(img[y: y + h, x: x + w], (width, heigth))

    cv2.imwrite(str(cropped_file), cropped_img)

    if faces.any():
        return 1
    else:
        notdetected.append(file_path.name)
        return 0

root_path = Path("Face Datasets/extended-cohn-kanade-images/cohn-kanade-images")
# if you use Windows comment next line
root_path = PurePosixPath(root_path)

cropped_path = Path("cropped/")
# if you use Windows comment next line
cropped_path = PurePosixPath(cropped_path)

#Specify size of the cropped images
width = 128
heigth = 128

#variables to calculate the efficiency
notdetected = []
detected = 0
withoutemotion = 0

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for f in sorted(listdir_nohidden(root_path)):
    for sf in sorted(listdir_nohidden(root_path/f)):
        files_names = [f for f in sorted(listdir_nohidden(root_path/f/sf))]
        if files_names[-1].endswith("emotion.txt"):
            file_path = root_path/f/sf/files_names[-2] # -2 because emotions are there
            file = open(str(root_path/f/sf/files_names[-1]), 'r')
            emotion = int(file.readline().strip()[0]) #read emotion label from file
            print("emotion label: " + str(emotion))
            labeled_file_name = file_path.name[:-4]+'_'+str(emotion)+file_path.name[-4:]
            cropped_labeled_path = Path(str(cropped_path) + '/' + labeled_file_name)
            # if you use Windows comment next line
            cropped_labeled_path = PurePosixPath(cropped_labeled_path)
            print(cropped_labeled_path)

            d = crop_detected_faces(file_path, cropped_labeled_path, faceCascade, notdetected)
            detected += d
        else:
            withoutemotion += 1

print("------------------------------------------------")
print("Successfully cropped", detected, "images")
print("Images with no emotion label:", withoutemotion)
print(len(notdetected), "images with no face detected")
print(*notdetected, sep='\n')
print("However, remember to check if all of the faces were detected properly!")
