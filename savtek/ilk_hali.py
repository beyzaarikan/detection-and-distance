import pyzed.sl as sl
import cv2
import numpy as np

MIN_AREA = 8000
coordinates=[]
distances=[]
barriers=[] 

# ZED kamera ayarları
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Kamera çözünürlüğünü ayarla
init_params.camera_fps = 30  # Kamera FPS'ini ayarla

init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units
runtime_parameters = sl.RuntimeParameters()

# ZED kamera nesnesini başlatq
zed = sl.Camera()

# Kamera açma işlemini başlat
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("ZED kamera açılamadı!")
    exit(1)

# Ana döngü
while True:
    # ZED kameradan bir kare alın
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Renk ve derinlik görüntülerini al        
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()

        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ) # Retrieve colored point cloud
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Sol renk görüntüsünü al
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # Derinlik haritasını al
        
        # Renk görüntüsünü OpenCV formatına dönüştür
        frame = image.get_data()

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv_img,(170, 50, 50), (180, 255, 255))
        mask2 = cv2.inRange(hsv_img,(0, 50, 50), (10, 255, 255))
        final_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(contour) for contour in contours]
        max_area_index = areas.index(max(areas))
        largest_contour = contours[max_area_index]
        for point in largest_contour:
            x, y = point[0]
            coordinates.append(point[0])
            #print("X:", x, "Y:", y)

        

        for contour in contours:
            epsilon = 0.01*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP (contour,epsilon, True )

            if cv2.contourArea(contour) > MIN_AREA:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M['m10'] / M['m00'])
                    center_y = int(M['m01'] / M['m00'])
                    weightCenter = (center_x, center_y)
                    cv2.circle(hsv_img, weightCenter, 5, (0, 0, 255), -1)

                    # KOSE KOORDİNATI (MAX_X,MAX_Y)
                    downLeftPoint=np.amax(approx,axis=0)
                    upRightPoint=np.amin(approx,axis=0)       
                    barriers.append([weightCenter, (downLeftPoint, upRightPoint)])
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(hsv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    for coordinate in coordinates:
                        depth_value = depth.get_value(coordinate[0], coordinate[1])
                        distances.append(depth_value)

                    if len(distances) > 0:
                        mean_distance = np.mean(distances)
                        print(f"Nesnenin ortalama uzaklığı: {mean_distance/100:.2f} cm")
                        if mean_distance<200:
                            print("sicak sicak sicak")
                        else:
                            print("soguk soguk soguk")
                    else:
                        print("Nesne tespit edilemedi.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

#cv2.imshow("Sonuç", hsv_img)

zed.close()

cv2.waitKey(0)
cv2.destroyAllWindows()




