import argparse

import cv2
import numpy as np
from imutils import paths
from matplotlib import font_manager
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


def paint_text(img, text, pos, color):
    """替代openCV無法畫中文的問題，使用PIL來將中文文字畫上"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    absolute_path = None
    for font_path in sorted(font_manager.findSystemFonts(fontpaths=None, fontext="ttf"), reverse=True):
        if "song" in font_path.lower():
            font_manager.get_font(font_path).get_path()
            absolute_path = font_path
            break

    if not absolute_path:
        # 假如沒有適合的字體，使用預設
        print("[INFO] use default font")
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(absolute_path, size=16)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    return img


def main():
    # 讓辨識結果有一點透明度
    ALPHA = 0.6

    # 根據輸入的圖片與語言開始辨識
    ocr = PaddleOCR(lang=args["lang"])
    if not args["image"].endswith("jpg"):
        img_path_list = paths.list_images(args["image"])
        for img_path in img_path_list:
            results = ocr.ocr(img=img_path)
            print(img_path)

            output_img = cv2.imread(img_path)
            copy_img = output_img.copy()
            # 依次處理辨識結果
            for result in results:
                box = np.array(result[0]).astype(np.int64)
                text = result[1][0]
                score = result[1][1]

                color = np.random.randint(0, high=256, size=(3,)).tolist()
                cv2.fillPoly(copy_img, [box], color)
                if args["lang"] == "en":
                    cv2.putText(
                        copy_img,
                        f"{text}({round(score, 2)})",
                        (box[1][0], box[1][1]),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.1,
                        (0, 255, 0),
                    )
                else:
                    copy_img = paint_text(copy_img, f"{text}({round(score, 2)})", (box[1][0], box[1][1]), (0, 255, 0))
            final = cv2.addWeighted(copy_img, ALPHA, output_img, 1 - ALPHA, 0)
            cv2.imshow("final result", final)
            cv2.waitKey(0)
        exit(0)
    results = ocr.ocr(img=args["image"])
    print(results)

    output_img = cv2.imread(args["image"])
    copy_img = output_img.copy()

    # 依次處理辨識結果
    for result in results:
        box = np.array(result[0]).astype(np.int64)
        text = result[1][0]
        score = result[1][1]

        color = np.random.randint(0, high=256, size=(3,)).tolist()
        cv2.fillPoly(copy_img, [box], color)
        if args["lang"] == "en":
            cv2.putText(
                copy_img,
                f"{text}({round(score, 2)})",
                (box[1][0], box[1][1]),
                cv2.FONT_HERSHEY_PLAIN,
                1.1,
                (0, 255, 0),
            )
        else:
            copy_img = paint_text(copy_img, f"{text}({round(score, 2)})", (box[1][0], box[1][1]), (0, 255, 0))
    final = cv2.addWeighted(copy_img, ALPHA, output_img, 1 - ALPHA, 0)
    cv2.imshow("final result", final)
    cv2.waitKey(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-i", "--image", required=True, help="你要辨識的圖片")
    ap.add_argument("-l", "--lang", default="en", type=str, help="辨識模型要使用何種語言")

    args = vars(ap.parse_args())

    main()
