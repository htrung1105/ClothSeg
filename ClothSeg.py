import cv2
import numpy as np
import os

def create_mask(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _ , tresh = cv2.threshold(gray,np.mean(gray), 255, cv2.THRESH_BINARY_INV   )

    # GET CONTOURS

    contours , hierarchy = cv2.findContours(tresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # lets get the bigger area

    cnt = sorted(contours, key=cv2.contourArea)[-1]

    mask = np.zeros( (1024, 768), dtype="uint8" )

    maskedRed = cv2.drawContours(mask,[cnt] , -1 , (0 , 0 , 255), -1)
    maskedFinal = cv2.drawContours(mask,[cnt] , -1 , (255 , 255 , 255), -1)
    
    return maskedFinal

def main():
    for filename in os.listdir("image"):
        img = cv2.imread(os.path.join("image", filename))
        mask = create_mask(img)
        cv2.imwrite(os.path.join("mask", filename[:-4] + "_mask.jpg"), mask)

main()

