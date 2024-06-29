import pyzed.sl as sl
import cv2
import numpy as np

class Detect_barrier:
    def __init__(self, min_area=8000):
        self.MIN_AREA = min_area
        self.barriers = []
        self.coordinates = []
        self.distances = []

        # ZED kamera ayarları
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720  # Kamera çözünürlüğünü ayarla
        self.init_params.camera_fps = 30  # Kamera FPS'ini ayarla
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Set the depth mode to performance (fastest)
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units
        self.runtime_parameters = sl.RuntimeParameters()

        # ZED kamera nesnesini başlat
        self.zed = sl.Camera()

    def open_camera(self):
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("ZED kamera açılamadı!")
            exit(1)

    def process_image(self, img_path):   # bu kısım resim pahti verirsen uygulanacak kısım
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

    def process_video(self):   #bu kısım ZED den gelen veriler icin
        self.open_camera()
        while True:
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                image = sl.Mat()
                depth = sl.Mat()
                point_cloud = sl.Mat()
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ) # Retrieve colored point cloud
                self.zed.retrieve_image(image, sl.VIEW.LEFT)  # Sol renk görüntüsünü al
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # Derinlik haritasını al
                
                frame = image.get_data()

                hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv_img,(170, 50, 50), (180, 255, 255))
                mask2 = cv2.inRange(hsv_img,(0, 50, 50), (10, 255, 255))
                final_mask = cv2.bitwise_or(mask1, mask2)               
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #bu kısım tespit edilen nesnenin tüm bölgesindeki koordinatları bir listeye atıyor
                areas = [cv2.contourArea(contour) for contour in contours]
                max_area_index = areas.index(max(areas))
                largest_contour = contours[max_area_index]
                for point in largest_contour:
                    x, y = point[0]
                    self.coordinates.append(point[0])
                
                if len(self.coordinates) > 200: #bu kısım listedeki son 200 ü alsın diye
                    self.coordinates = self.coordinates[-200:]

                self.detect_objects(contours, frame)
                self.calculate_distance(depth,self.coordinates)

                cv2.imshow("Orijinal görüntü", frame)
                cv2.imshow("Sonuç", final_mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.zed.close()
        cv2.destroyAllWindows()

    def detect_objects(self, contours, frame):
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if cv2.contourArea(contour) > self.MIN_AREA:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M['m10'] / M['m00'])
                    center_y = int(M['m01'] / M['m00'])
                    weightCenter = (center_x, center_y)

                    cv2.circle(frame, weightCenter, 5, (0, 0, 255), -1)

                    downLeftPoint = np.amax(approx, axis=0)
                    upRightPoint = np.amin(approx, axis=0)

                    self.barriers.append([weightCenter, (downLeftPoint, upRightPoint)])
                    #print(self.barriers)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def calculate_distance(self, depth,coordinates):
        for coordinate in coordinates:
            depth_value = depth.get_value(coordinate[0], coordinate[1])
            self.distances.append(depth_value)

        if len(self.distances) > 0:
            mean_distance = np.mean(self.distances)
            print(f"Nesnenin ortalama uzaklığı: {mean_distance/1000:.2f} m")

            '''
            if mean_distance<2:
                return True
            else:
                return False

            '''
            
                   

def main():
    processor = Detect_barrier()
    
    # görüntü işleme
    #processor.process_image("den.png")
    
    # ZED kamera işleme
    processor.process_video()

if __name__ == "__main__":
    main()
