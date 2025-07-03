import cv2

# Load Haar cascade
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load image
image = cv2.imread('WIN_20250702_11_23_41_Pro.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_haar_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)

# Show image
cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
