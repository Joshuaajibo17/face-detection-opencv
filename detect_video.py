import cv2

# Open webcam (0 = default camera)
capture = cv2.VideoCapture(0)

# Load Haar cascade
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    success, image = capture.read()
    if not success:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_haar_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Live Face Detection", image)

    # Press ESC to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Cleanup
capture.release()
cv2.destroyAllWindows()
