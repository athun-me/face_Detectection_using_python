import cv2
image_path = "profesor.jpg"
xml_path = "raw.xml.xml"

j = cv2.CascadeClassifier(xml_path)
img = cv2.imread(image_path)

grey_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_detect = j.detectMultiScale(
    grey_color,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(20, 20),
    flags = cv2.CASCADE_SCALE_IMAGE

)
print("found {0} face".format(len(face_detect)))

for (x, y, w, h) in face_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Face Detected", img)
cv2.waitKey(0)

