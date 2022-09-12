import argparse

import cv2
import imutils
from paddleocr import PaddleOCR


def main():
    ocr = PaddleOCR(lang="en")

    # 你的圖片從args["image"]傳入後，轉成灰階圖片(gray)
    img = cv2.imread(args["image"])
    cv2.imshow("Origin", img)
    cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    ### 增強黑底白字區塊的對比
    # 定義與車牌類似比例的kernel (licenseKern)，這裡的(15,5)不可太小避免無法有效突顯；不可太大避免突顯區塊太多而失去焦點
    licenseKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    # 黑帽處理，突顯背景是黑色、前景是白色的區塊
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, licenseKern)
    cv2.imshow("Blackhat", blackhat)
    cv2.waitKey(0)

    ### 找出區塊的邊緣部分
    # 分別針對水平方向(X)與垂直方向(Y)處理後，再用權重相加。避免直接用一行dx=1,dy=1有"相加抵消"的問題而不是我們想要的結果
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradX = cv2.convertScaleAbs(gradX)
    gradY = cv2.convertScaleAbs(gradY)
    Sobel = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
    cv2.imshow("Sobel", Sobel)
    cv2.waitKey(0)

    ### 合併白色區塊
    # 利用模糊化處理太清楚的邊綠，
    # 再透過閉合處理把白色區塊中太小的黑色都去除，
    # 最後再進行二值化處理
    Sobel = cv2.GaussianBlur(Sobel, (15, 15), 0)
    Sobel = cv2.morphologyEx(Sobel, cv2.MORPH_CLOSE, licenseKern)
    thresh = cv2.threshold(Sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Grad Thresh", thresh)
    cv2.waitKey(0)

    ### 第一段去雜訊
    # 先擴大再縮小 (針對白色是前景而言；如果是黑色才是想要的區塊，則下面兩行順序相反)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)
    cv2.imshow("Dilate/Erode", thresh)
    cv2.waitKey(0)

    ### 第二段去雜訊
    # 用更大的kernel size來作閉合處理，找出大片白色區塊並抹去其中的小黑色區塊
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Light Regions", light)
    cv2.waitKey(0)

    # 以前一步驟的結果作為遮罩，與前面第一段的結果作AND操作來得到最後結果
    final = cv2.bitwise_and(thresh, thresh, mask=light)
    cv2.imshow("Final", final)
    cv2.waitKey(0)

    # 找出白色區塊的輪廓，再由面積大到小排序
    cnts = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    import numpy as np

    gg = np.dstack([final, final, final])
    cv2.drawContours(gg, cnts, -1, (0, 0, 255), 3)
    cv2.imshow("Contours", gg)
    cv2.waitKey(0)
    exit(0)
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
