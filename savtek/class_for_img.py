import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, min_area=8000):
        self.MIN_AREA = min_area
        self.barriers = []

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv_img,(170, 50, 50), (180, 255, 255))
        mask2 = cv2.inRange(hsv_img,(0, 50, 50), (10, 255, 255))
        final_mask = cv2.bitwise_or(mask1, mask2)    

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.detect_objects(contours, img)

        cv2.imshow("Orijinal görüntü", img)
        cv2.imshow("Sonuç", final_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_objects(self, contours, img):
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if cv2.contourArea(contour) > self.MIN_AREA:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M['m10'] / M['m00'])
                    center_y = int(M['m01'] / M['m00'])
                    weightCenter = (center_x, center_y)

                    cv2.circle(img, weightCenter, 5, (0, 0, 255), -1)

                    downLeftPoint = np.amax(approx, axis=0)
                    upRightPoint = np.amin(approx, axis=0)

                    self.barriers.append([weightCenter, (downLeftPoint, upRightPoint)])
                    print(self.barriers)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main():
    processor = ImageProcessor()
    processor.process_image("den.png")

if __name__ == "__main__":
    main()
