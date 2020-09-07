import numpy as np
import cv2
import tensorflow

from ssd_detector.ssd_detector import SSDDetector

img = cv2.imread('1.jpg')

model_path = 'mobilenetv2_ssd.pb'
ssd_detector = SSDDetector(det_threshold = .3, model_path = model_path)

pred = ssd_detector.predict(img)
print(pred)

x1, y1, x2, y2 = pred[0][0]
color = (100, 100, 255)
thickness = 1
cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
cv2.imshow('img', img)
cv2.waitKey(0)
