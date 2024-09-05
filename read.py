import cv2 as cv

img = cv.imread(r'C:\Users\91952\Desktop\open cv\photos\cat.jpeg')

cv.imshow('cat', img)

cv.waitKey(0)

resized_img = rescaleFrame(img)
cv.imshow('Image', resized_img)