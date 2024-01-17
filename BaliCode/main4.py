import cv2
from deepface import DeepFace

img = cv2.imread("faces/img13.jpg")

results = DeepFace.analyze(img, actions=("gender", "age"))

print(results)

