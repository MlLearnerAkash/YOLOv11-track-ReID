from ultralytics import YOLO, solutions
import cv2
import logging

weights="/mnt/data/weights/all_data_v11/yolov11_all_data_02122024_5/weights/best.pt"
model = YOLO(weights)
source = '/mnt/data/demo_video/surg1_needle_1.avi'
view_img=False
line_thickness=2
line_points = [(800, 1280), (800, 1870), (1400, 1870), (1400, 1280)]


#TODO:
#BB comparison:
#1. Check if the BB is inside the polygon
### give the look up table for the class names.
class bbox_is_inside():

    def IS_inside(self,bbox):
        try:
            is_inside  = False
            is_outside = False
            bbox_int = [int(x) for x in bbox]

            roi_x1,roi_y1,roi_x2,roi_y2 = [800, 1280, 1400, 1870]
            x1,y1,x2,y2 = bbox_int

            # Function to check if a point is inside the ROI
            def is_inside_roi(x, y, roi_x1, roi_y1, roi_x2, roi_y2):
                return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2

            # Check if any of the points are inside the ROI
            if (is_inside_roi(x1, y1, roi_x1, roi_y1, roi_x2, roi_y2) or
                is_inside_roi(x2, y2, roi_x1, roi_y1, roi_x2, roi_y2)) or (is_inside_roi(x1, y2, roi_x1, roi_y1, roi_x2, roi_y2))or (is_inside_roi(x2, y1, roi_x1, roi_y1, roi_x2, roi_y2)):
                is_inside = True
            else:
                is_outside = True
            return is_inside, is_outside
        except Exception as e:
            logging.error(f'Error in IS_inside function: {e}')

    def output_configure(self,list1,list2):
        try:
            look_up = {3:'a',0:'b',6:'c',5:'d'}  #change as per your look up of class names
            list2 = [look_up[int(i)] for i in list2 if int(i) in look_up]
            zipped_list = list(zip(list1, list2))
            final_output = []
            for bbox,cls_name in zipped_list:
                is_inside,is_outside = self.IS_inside(bbox)
                temp = {'class_name':cls_name,'is_inside':is_inside,'is_outside':is_outside}
                final_output.append(temp)
            return final_output
        except Exception as e:
            logging.error(f'Error in output_configure function: {e}')
        

# x = bbox_is_inside()
# print(x.output_configure(list1,list2))



cap = cv2.VideoCapture(source)
while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break
    res = model.predict(im0, save=False, imgsz = 2048)
    bbox = res[0].boxes.xyxy.tolist()
    cls = res[0].boxes.cls.tolist()
    x = bbox_is_inside()
    print(x.output_configure(bbox,cls))
    






