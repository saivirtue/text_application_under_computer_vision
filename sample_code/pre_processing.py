import argparse

import cv2
import imutils
import numpy as np
from paddleocr import PaddleOCR


def main():
    ocr = PaddleOCR(lang="en")

    img = cv2.imread(args["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    licenseKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, licenseKern)
    cv2.imshow("Blackhat", blackhat)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradX = cv2.convertScaleAbs(gradX)
    gradY = cv2.convertScaleAbs(gradY)
    Sobel = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
    cv2.imshow("Sobel", Sobel)

    Sobel = cv2.GaussianBlur(Sobel, (15, 15), 0)
    Sobel = cv2.morphologyEx(Sobel, cv2.MORPH_CLOSE, licenseKern)
    thresh = cv2.threshold(Sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Grad Thresh", thresh)
    cv2.waitKey(0)
    exit(0)

    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)
    cv2.imshow("Grad Erode/Dilate", thresh)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Light Regions", light)

    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    cv2.imshow("final", thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        ar = w / h * 1.0
        if 4 <= len(approx) <= 5 and ar > 2 and area > 1000:
            print(f"area: {area}, ar: {ar}, len:{len(approx)}")

            results = ocr.ocr(img=img[y : y + h, x : x + w])
            print(results)
            # cv2.drawContours(img, [c], 0, (255, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="the image path")
    args = vars(ap.parse_args())
    main()
