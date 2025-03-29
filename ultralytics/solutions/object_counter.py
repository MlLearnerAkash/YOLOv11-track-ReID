# # Ultralytics YOLO 🚀, AGPL-3.0 license

# from ultralytics.solutions.solutions import BaseSolution
# from ultralytics.utils.plotting import Annotator, colors


# class ObjectCounter(BaseSolution):
#     """
#     A class to manage the counting of objects in a real-time video stream based on their tracks.

#     This class extends the BaseSolution class and provides functionality for counting objects moving in and out of a
#     specified region in a video stream. It supports both polygonal and linear regions for counting.

#     Attributes:
#         in_count (int): Counter for objects moving inward.
#         out_count (int): Counter for objects moving outward.
#         counted_ids (List[int]): List of IDs of objects that have been counted.
#         classwise_counts (Dict[str, Dict[str, int]]): Dictionary for counts, categorized by object class.
#         region_initialized (bool): Flag indicating whether the counting region has been initialized.
#         show_in (bool): Flag to control display of inward count.
#         show_out (bool): Flag to control display of outward count.

#     Methods:
#         count_objects: Counts objects within a polygonal or linear region.
#         store_classwise_counts: Initializes class-wise counts if not already present.
#         display_counts: Displays object counts on the frame.
#         count: Processes input data (frames or object tracks) and updates counts.

#     Examples:
#         >>> counter = ObjectCounter()
#         >>> frame = cv2.imread("frame.jpg")
#         >>> processed_frame = counter.count(frame)
#         >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
#     """

#     def __init__(self, **kwargs):
#         """Initializes the ObjectCounter class for real-time object counting in video streams."""
#         super().__init__(**kwargs)

#         self.in_count = 0  # Counter for objects moving inward
#         self.out_count = 0  # Counter for objects moving outward
#         self.counted_ids = []  # List of IDs of objects that have been counted
#         self.classwise_counts = {}  # Dictionary for counts, categorized by object class
#         self.region_initialized = False  # Bool variable for region initialization

#         self.show_in = self.CFG["show_in"]
#         self.show_out = self.CFG["show_out"]

#     def count_objects(self, current_centroid, track_id, prev_position, cls):
#         """
#         Counts objects within a polygonal or linear region based on their tracks.

#         Args:
#             current_centroid (Tuple[float, float]): Current centroid values in the current frame.
#             track_id (int): Unique identifier for the tracked object.
#             prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track.
#             cls (int): Class index for classwise count updates.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
#             >>> box = [130, 230, 150, 250]
#             >>> track_id = 1
#             >>> prev_position = (120, 220)
#             >>> cls = 0
#             >>> counter.count_objects(current_centroid, track_id, prev_position, cls)
#         """
#         if prev_position is None or track_id in self.counted_ids:
#             return

#         if len(self.region) == 2:  # Linear region (defined as a line segment)
#             line = self.LineString(self.region)  # Check if the line intersects the trajectory of the object
#             if line.intersects(self.LineString([prev_position, current_centroid])):
#                 # Determine orientation of the region (vertical or horizontal)
#                 if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
#                     # Vertical region: Compare x-coordinates to determine direction
#                     if current_centroid[0] > prev_position[0]:  # Moving right
#                         self.in_count += 1
#                         self.classwise_counts[self.names[cls]]["IN"] += 1
#                     else:  # Moving left
#                         self.out_count += 1
#                         self.classwise_counts[self.names[cls]]["OUT"] += 1
#                 # Horizontal region: Compare y-coordinates to determine direction
#                 elif current_centroid[1] > prev_position[1]:  # Moving downward
#                     self.in_count += 1
#                     self.classwise_counts[self.names[cls]]["IN"] += 1
#                 else:  # Moving upward
#                     self.out_count += 1
#                     self.classwise_counts[self.names[cls]]["OUT"] += 1
#                 self.counted_ids.append(track_id)

#         elif len(self.region) > 2:  # Polygonal region
#             polygon = self.Polygon(self.region)
#             if polygon.contains(self.Point(current_centroid)):
#                 # Determine motion direction for vertical or horizontal polygons
#                 region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
#                 region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

#                 if (
#                     region_width < region_height
#                     and current_centroid[0] > prev_position[0]
#                     or region_width >= region_height
#                     and current_centroid[1] > prev_position[1]
#                 ):  # Moving right
#                     self.in_count += 1
#                     self.classwise_counts[self.names[cls]]["IN"] += 1
#                 else:  # Moving left
#                     self.out_count += 1
#                     self.classwise_counts[self.names[cls]]["OUT"] += 1
#                 self.counted_ids.append(track_id)

#     def store_classwise_counts(self, cls):
#         """
#         Initialize class-wise counts for a specific object class if not already present.

#         Args:
#             cls (int): Class index for classwise count updates.

#         This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
#         initializing 'IN' and 'OUT' counts to zero if the class is not already present.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> counter.store_classwise_counts(0)  # Initialize counts for class index 0
#             >>> print(counter.classwise_counts)
#             {'person': {'IN': 0, 'OUT': 0}}
#         """
#         if self.names[cls] not in self.classwise_counts:
#             self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

#     def display_counts(self, im0):
#         """
#         Displays object counts on the input image or frame.

#         Args:
#             im0 (numpy.ndarray): The input image or frame to display counts on.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> frame = cv2.imread("image.jpg")
#             >>> counter.display_counts(frame)
#         """
#         labels_dict = {
#             str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
#             f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
#             for key, value in self.classwise_counts.items()
#             if value["IN"] != 0 or value["OUT"] != 0
#         }

#         if labels_dict:
#             self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

#     def count(self, im0):
#         """
#         Processes input data (frames or object tracks) and updates object counts.

#         This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
#         object counts, and displays the results on the input image.

#         Args:
#             im0 (numpy.ndarray): The input image or frame to be processed.

#         Returns:
#             (numpy.ndarray): The processed image with annotations and count information.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> frame = cv2.imread("path/to/image.jpg")
#             >>> processed_frame = counter.count(frame)
#         """
#         if not self.region_initialized:
#             self.initialize_region()
#             self.region_initialized = True

#         self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
#         self.extract_tracks(im0)  # Extract tracks

#         self.annotator.draw_region(
#             reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
#         )  # Draw region

#         # Iterate over bounding boxes, track ids and classes index
#         for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
#             # Draw bounding box and counting region
#             self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
#             self.store_tracking_history(track_id, box)  # Store track history
#             self.store_classwise_counts(cls)  # store classwise counts in dict

#             # Draw tracks of objects
#             self.annotator.draw_centroid_and_tracks(
#                 self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
#             )
#             current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
#             # store previous position of track for object counting
#             prev_position = None
#             if len(self.track_history[track_id]) > 1:
#                 prev_position = self.track_history[track_id][-2]
#             self.count_objects(current_centroid, track_id, prev_position, cls)  # Perform object counting

#         self.display_counts(im0)  # Display the counts on the frame
#         self.display_output(im0)  # display output with base class function

#         return im0  # return output image for more usage

# # Ultralytics YOLO 🚀, AGPL-3.0 license

# from ultralytics.solutions.solutions import BaseSolution
# from ultralytics.utils.plotting import Annotator, colors


# class ObjectCounter(BaseSolution):
#     """
#     A class to manage the counting of objects in a real-time video stream based on their tracks.

#     This class extends the BaseSolution class and provides functionality for counting objects moving in and out of a
#     specified region in a video stream. It supports both polygonal and linear regions for counting.

#     Attributes:
#         in_count (int): Counter for objects moving inward.
#         out_count (int): Counter for objects moving outward.
#         counted_ids (List[int]): List of IDs of objects that have been counted.
#         classwise_counts (Dict[str, Dict[str, int]]): Dictionary for counts, categorized by object class.
#         region_initialized (bool): Flag indicating whether the counting region has been initialized.
#         show_in (bool): Flag to control display of inward count.
#         show_out (bool): Flag to control display of outward count.

#     Methods:
#         count_objects: Counts objects within a polygonal or linear region.
#         store_classwise_counts: Initializes class-wise counts if not already present.
#         display_counts: Displays object counts on the frame.
#         count: Processes input data (frames or object tracks) and updates counts.

#     Examples:
#         >>> counter = ObjectCounter()
#         >>> frame = cv2.imread("frame.jpg")
#         >>> processed_frame = counter.count(frame)
#         >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
#     """

#     def __init__(self, **kwargs):
#         """Initializes the ObjectCounter class for real-time object counting in video streams."""
#         super().__init__(**kwargs)

#         self.in_count = 0  # Counter for objects moving inward
#         self.out_count = 0  # Counter for objects moving outward
#         self.counted_ids = []  # List of IDs of objects that have been counted
#         self.classwise_counts = {}  # Dictionary for counts, categorized by object class
#         self.region_initialized = False  # Bool variable for region initialization

#         self.show_in = self.CFG["show_in"]
#         self.show_out = self.CFG["show_out"]

#     def count_objects(self, current_centroid, track_id, prev_position, cls):
#         """
#         Counts objects within a polygonal or linear region based on their tracks.

#         Args:
#             current_centroid (Tuple[float, float]): Current centroid values in the current frame.
#             track_id (int): Unique identifier for the tracked object.
#             prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track.
#             cls (int): Class index for classwise count updates.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
#             >>> box = [130, 230, 150, 250]
#             >>> track_id = 1
#             >>> prev_position = (120, 220)
#             >>> cls = 0
#             >>> counter.count_objects(current_centroid, track_id, prev_position, cls)
#         """
#         if prev_position is None or track_id in self.counted_ids:
#             return

#         if len(self.region) == 2:  # Linear region (defined as a line segment)
#             line = self.LineString(self.region)  # Check if the line intersects the trajectory of the object
#             if line.intersects(self.LineString([prev_position, current_centroid])):
#                 # Determine orientation of the region (vertical or horizontal)
#                 if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
#                     # Vertical region: Compare x-coordinates to determine direction
#                     if current_centroid[0] > prev_position[0]:  # Moving right
#                         self.in_count += 1
#                         self.classwise_counts[self.names[cls]]["IN"] += 1
#                     else:  # Moving left
#                         self.out_count += 1
#                         self.classwise_counts[self.names[cls]]["OUT"] += 1
#                 # Horizontal region: Compare y-coordinates to determine direction
#                 elif current_centroid[1] > prev_position[1]:  # Moving downward
#                     self.in_count += 1
#                     self.classwise_counts[self.names[cls]]["IN"] += 1
#                 else:  # Moving upward
#                     self.out_count += 1
#                     self.classwise_counts[self.names[cls]]["OUT"] += 1
#                 self.counted_ids.append(track_id)

#         elif len(self.region) > 2:  # Polygonal region
#             polygon = self.Polygon(self.region)
#             if polygon.contains(self.Point(current_centroid)):
#                 # Determine motion direction for vertical or horizontal polygons
#                 region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
#                 region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

#                 if (
#                     region_width < region_height
#                     and current_centroid[0] > prev_position[0]
#                     or region_width >= region_height
#                     and current_centroid[1] > prev_position[1]
#                 ):  # Moving right
#                     self.in_count += 1
#                     self.classwise_counts[self.names[cls]]["IN"] += 1
#                 else:  # Moving left
#                     self.out_count += 1
#                     self.classwise_counts[self.names[cls]]["OUT"] += 1
#                 self.counted_ids.append(track_id)

#     def store_classwise_counts(self, cls):
#         """
#         Initialize class-wise counts for a specific object class if not already present.

#         Args:
#             cls (int): Class index for classwise count updates.

#         This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
#         initializing 'IN' and 'OUT' counts to zero if the class is not already present.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> counter.store_classwise_counts(0)  # Initialize counts for class index 0
#             >>> print(counter.classwise_counts)
#             {'person': {'IN': 0, 'OUT': 0}}
#         """
#         if self.names[cls] not in self.classwise_counts:
#             self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

#     def display_counts(self, im0):
#         """
#         Displays object counts on the input image or frame.

#         Args:
#             im0 (numpy.ndarray): The input image or frame to display counts on.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> frame = cv2.imread("image.jpg")
#             >>> counter.display_counts(frame)
#         """
#         labels_dict = {
#             str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
#             f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
#             for key, value in self.classwise_counts.items()
#             if value["IN"] != 0 or value["OUT"] != 0
#         }

#         if labels_dict:
#             self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

#     def count(self, im0):
#         """
#         Processes input data (frames or object tracks) and updates object counts.

#         This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
#         object counts, and displays the results on the input image.

#         Args:
#             im0 (numpy.ndarray): The input image or frame to be processed.

#         Returns:
#             (numpy.ndarray): The processed image with annotations and count information.

#         Examples:
#             >>> counter = ObjectCounter()
#             >>> frame = cv2.imread("path/to/image.jpg")
#             >>> processed_frame = counter.count(frame)
#         """
#         if not self.region_initialized:
#             self.initialize_region()
#             self.region_initialized = True

#         self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
#         self.extract_tracks(im0)  # Extract tracks

#         self.annotator.draw_region(
#             reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
#         )  # Draw region

#         # Iterate over bounding boxes, track ids and classes index
#         for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
#             # Draw bounding box and counting region
#             self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
#             self.store_tracking_history(track_id, box)  # Store track history
#             self.store_classwise_counts(cls)  # store classwise counts in dict

#             # Draw tracks of objects
#             self.annotator.draw_centroid_and_tracks(
#                 self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
#             )
#             current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
#             # store previous position of track for object counting
#             prev_position = None
#             if len(self.track_history[track_id]) > 1:
#                 prev_position = self.track_history[track_id][-2]
#             self.count_objects(current_centroid, track_id, prev_position, cls)  # Perform object counting

#         self.display_counts(im0)  # Display the counts on the frame
#         self.display_output(im0)  # display output with base class function

#         return im0  # return output image for more usage

#NOTE: Added
# # Ultralytics YOLO 🚀, AGPL-3.0 license

# from collections import defaultdict

# import cv2

# from ultralytics.utils.checks import check_imshow, check_requirements
# from ultralytics.utils.plotting import Annotator, colors

# check_requirements("shapely>=2.0.0")

# from shapely.geometry import LineString, Point, Polygon


# class ObjectCounter:
#     """A class to manage the counting of objects in a real-time video stream based on their tracks."""

#     def __init__(
#         self,
#         classes_names,
#         reg_pts=None,
#         count_reg_color=(255, 0, 255),
#         count_txt_color=(0, 0, 0),
#         count_bg_color=(255, 255, 255),
#         line_thickness=2,
#         track_thickness=2,
#         view_img=False,
#         view_in_counts=True,
#         view_out_counts=True,
#         draw_tracks=False,
#         track_color=None,
#         region_thickness=5,
#         line_dist_thresh=15,
#         cls_txtdisplay_gap=50,
#     ):
#         """
#         Initializes the ObjectCounter with various tracking and counting parameters.

#         Args:
#             classes_names (dict): Dictionary of class names.
#             reg_pts (list): List of points defining the counting region.
#             count_reg_color (tuple): RGB color of the counting region.
#             count_txt_color (tuple): RGB color of the count text.
#             count_bg_color (tuple): RGB color of the count text background.
#             line_thickness (int): Line thickness for bounding boxes.
#             track_thickness (int): Thickness of the track lines.
#             view_img (bool): Flag to control whether to display the video stream.
#             view_in_counts (bool): Flag to control whether to display the in counts on the video stream.
#             view_out_counts (bool): Flag to control whether to display the out counts on the video stream.
#             draw_tracks (bool): Flag to control whether to draw the object tracks.
#             track_color (tuple): RGB color of the tracks.
#             region_thickness (int): Thickness of the object counting region.
#             line_dist_thresh (int): Euclidean distance threshold for line counter.
#             cls_txtdisplay_gap (int): Display gap between each class count.
#         """

#         # Mouse events
#         self.is_drawing = False
#         self.selected_point = None

#         # Region & Line Information
#         self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts
#         self.line_dist_thresh = line_dist_thresh
#         self.counting_region = None
#         self.region_color = count_reg_color
#         self.region_thickness = region_thickness

#         # Image and annotation Information
#         self.im0 = None
#         self.tf = line_thickness
#         self.view_img = view_img
#         self.view_in_counts = view_in_counts
#         self.view_out_counts = view_out_counts

#         self.names = classes_names  # Classes names
#         self.annotator = None  # Annotator
#         self.window_name = "Ultralytics YOLOv8 Object Counter"

#         # Object counting Information
#         self.in_counts = 0
#         self.out_counts = 0
#         self.count_ids = []
#         self.class_wise_count = {}
#         self.count_txt_thickness = 0
#         self.count_txt_color = count_txt_color
#         self.count_bg_color = count_bg_color
#         self.cls_txtdisplay_gap = cls_txtdisplay_gap
#         self.fontsize = 0.6

#         # Tracks info
#         self.track_history = defaultdict(list)
#         self.track_thickness = track_thickness
#         self.draw_tracks = draw_tracks
#         self.track_color = track_color

#         # Check if environment supports imshow
#         self.env_check = check_imshow(warn=True)

#         # Initialize counting region
#         if len(self.reg_pts) == 2:
#             print("Line Counter Initiated.")
#             self.counting_region = LineString(self.reg_pts)
#         elif len(self.reg_pts) >= 3:
#             print("Polygon Counter Initiated.")
#             self.counting_region = Polygon(self.reg_pts)
#         else:
#             print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
#             print("Using Line Counter Now")
#             self.counting_region = LineString(self.reg_pts)

#     def mouse_event_for_region(self, event, x, y, flags, params):
#         """
#         Handles mouse events for defining and moving the counting region in a real-time video stream.

#         Args:
#             event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
#             x (int): The x-coordinate of the mouse pointer.
#             y (int): The y-coordinate of the mouse pointer.
#             flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
#             params (dict): Additional parameters for the function.
#         """
#         if event == cv2.EVENT_LBUTTONDOWN:
#             for i, point in enumerate(self.reg_pts):
#                 if (
#                     isinstance(point, (tuple, list))
#                     and len(point) >= 2
#                     and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
#                 ):
#                     self.selected_point = i
#                     self.is_drawing = True
#                     break

#         elif event == cv2.EVENT_MOUSEMOVE:
#             if self.is_drawing and self.selected_point is not None:
#                 self.reg_pts[self.selected_point] = (x, y)
#                 self.counting_region = Polygon(self.reg_pts)

#         elif event == cv2.EVENT_LBUTTONUP:
#             self.is_drawing = False
#             self.selected_point = None

#     # def extract_and_process_tracks(self, tracks):
#     #     """Extracts and processes tracks for object counting in a video stream."""
#     #     self.items_status = []
#     #     # Annotator Init and region drawing
#     #     self.annotator = Annotator(self.im0, self.tf, self.names)

#     #     # Draw region or line
#     #     self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

#     #     if tracks[0].boxes.id is not None:
#     #         boxes = tracks[0].boxes.xyxy.cpu()
#     #         clss = tracks[0].boxes.cls.cpu().tolist()
#     #         track_ids = tracks[0].boxes.id.int().cpu().tolist()
            
#     #         self.item_status= {}
#     #         # Extract tracks
#     #         for box, track_id, cls in zip(boxes, track_ids, clss):
#     #             self.item_status["class_id"]= cls
#     #             self.item_status["track_id"] = track_id
#     #             # Draw bounding box
#     #             self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

#     #             # Store class info
#     #             if self.names[cls] not in self.class_wise_count:
#     #                 self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

#     #             # Draw Tracks
#     #             track_line = self.track_history[track_id]
#     #             track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
#     #             if len(track_line) > 30:
#     #                 track_line.pop(0)

#     #             # Draw track trails
#     #             if self.draw_tracks:
#     #                 self.annotator.draw_centroid_and_tracks(
#     #                     track_line,
#     #                     color=self.track_color if self.track_color else colors(int(track_id), True),
#     #                     track_thickness=self.track_thickness,
#     #                 )

#     #             prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

#     #             # Count objects in any polygon
#     #             if len(self.reg_pts) >= 3:
#     #                 is_inside = self.counting_region.contains(Point(track_line[-1]))
#     #                 self.item_status["status"] = is_inside

#     #                 if prev_position is not None and is_inside and track_id not in self.count_ids:
#     #                     self.count_ids.append(track_id)

#     #                     if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
#     #                         self.in_counts += 1
#     #                         self.class_wise_count[self.names[cls]]["IN"] += 1
#     #                     else:
#     #                         self.out_counts += 1
#     #                         self.class_wise_count[self.names[cls]]["OUT"] += 1

#     #             # Count objects using line
#     #             elif len(self.reg_pts) == 2:
#     #                 if prev_position is not None and track_id not in self.count_ids:
#     #                     distance = Point(track_line[-1]).distance(self.counting_region)
#     #                     if distance < self.line_dist_thresh and track_id not in self.count_ids:
#     #                         self.count_ids.append(track_id)

#     #                         if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
#     #                             self.in_counts += 1
#     #                             self.class_wise_count[self.names[cls]]["IN"] += 1
#     #                         else:
#     #                             self.out_counts += 1
#     #                             self.class_wise_count[self.names[cls]]["OUT"] += 1
#     #             self.items_status.append(self.item_status)
#     #     labels_dict = {}

#     #     for key, value in self.class_wise_count.items():
#     #         if value["IN"] != 0 or value["OUT"] != 0:
#     #             if not self.view_in_counts and not self.view_out_counts:
#     #                 continue
#     #             elif not self.view_in_counts:
#     #                 labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
#     #             elif not self.view_out_counts:
#     #                 labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
#     #             else:
#     #                 labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

#     #     if labels_dict:
#     #         self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)

#     #     return self.items_status
#     def extract_and_process_tracks(self, tracks):
#         """Extracts and processes tracks for object counting in a video stream."""
#         self.items_status = []
#         # Annotator Init and region drawing
#         self.annotator = Annotator(self.im0, self.tf, self.names)

#         # Draw region or line
#         self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

#         if tracks[0].boxes.id is not None:
#             boxes = tracks[0].boxes.xyxy.cpu()
#             clss = tracks[0].boxes.cls.cpu().tolist()
#             track_ids = tracks[0].boxes.id.int().cpu().tolist()

#             # Extract tracks
#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 item_status = {"class_id": self.names[int(cls)], "track_id": track_id}  # Reset for each track
                
#                 # Draw bounding box
#                 self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

#                 # Initialize class info if not present
#                 if self.names[cls] not in self.class_wise_count:
#                     self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

#                 # Draw Tracks
#                 track_line = self.track_history[track_id]
#                 track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
#                 if len(track_line) > 30:
#                     track_line.pop(0)

#                 # Draw track trails
#                 if self.draw_tracks:
#                     self.annotator.draw_centroid_and_tracks(
#                         track_line,
#                         color=self.track_color if self.track_color else colors(int(track_id), True),
#                         track_thickness=self.track_thickness,
#                     )

#                 prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

#                 # Count objects in any polygon region
#                 is_inside = False
#                 if len(self.reg_pts) >= 3:
#                     is_inside = self.counting_region.contains(Point(track_line[-1]))
#                     item_status["is_inside"] = is_inside  # Update the inside status

#                     if prev_position is not None and is_inside and track_id not in self.count_ids:
#                         self.count_ids.append(track_id)

#                         if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
#                             self.in_counts += 1
#                             self.class_wise_count[self.names[cls]]["IN"] += 1
#                         else:
#                             self.out_counts += 1
#                             self.class_wise_count[self.names[cls]]["OUT"] += 1

#                 # Count objects crossing a line
#                 elif len(self.reg_pts) == 2:
#                     if prev_position is not None and track_id not in self.count_ids:
#                         distance = Point(track_line[-1]).distance(self.counting_region)
#                         if distance < self.line_dist_thresh and track_id not in self.count_ids:
#                             self.count_ids.append(track_id)

#                             if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
#                                 self.in_counts += 1
#                                 self.class_wise_count[self.names[cls]]["IN"] += 1
#                             else:
#                                 self.out_counts += 1
#                                 self.class_wise_count[self.names[cls]]["OUT"] += 1
#                 else:
#                     item_status["is_inside"] = is_inside  # Default if not in polygon or line-based counting

#                 # Append item_status for each track
#                 self.items_status.append(item_status)

#         labels_dict = {}
#         for key, value in self.class_wise_count.items():
#             if value["IN"] != 0 or value["OUT"] != 0:
#                 if not self.view_in_counts and not self.view_out_counts:
#                     continue
#                 elif not self.view_in_counts:
#                     labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
#                 elif not self.view_out_counts:
#                     labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
#                 else:
#                     labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

#         if labels_dict:
#             self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)

#         return self.items_status


#     def display_frames(self):
#         """Displays the current frame with annotations and regions in a window."""
#         if self.env_check:
#             cv2.namedWindow(self.window_name)
#             if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
#                 cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
#             cv2.imshow(self.window_name, self.im0)
#             # Break Window
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 return

#     def start_counting(self, im0, tracks):
#         """
#         Main function to start the object counting process.

#         Args:
#             im0 (ndarray): Current frame from the video stream.
#             tracks (list): List of tracks obtained from the object tracking process.
#         """
#         self.im0 = im0  # store image
#         self.item_status = self.extract_and_process_tracks(tracks)  # draw region even if no objects

#         if self.view_img:
#             self.display_frames()
#         return self.im0, self.item_status


# if __name__ == "__main__":
#     classes_names = {0: "person", 1: "car"}  # example class names
#     ObjectCounter(classes_names)

# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        classes_names,
        reg_pts=None,
        count_reg_color=(0, 0, 255),
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        """
        Initializes the ObjectCounter with various tracking and counting parameters.

        Args:
            classes_names (dict): Dictionary of class names.
            reg_pts (list): List of points defining the counting region.
            count_reg_color (tuple): RGB color of the counting region.
            count_txt_color (tuple): RGB color of the count text.
            count_bg_color (tuple): RGB color of the count text background.
            line_thickness (int): Line thickness for bounding boxes.
            track_thickness (int): Thickness of the track lines.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the in counts on the video stream.
            view_out_counts (bool): Flag to control whether to display the out counts on the video stream.
            draw_tracks (bool): Flag to control whether to draw the object tracks.
            track_color (tuple): RGB color of the tracks.
            region_thickness (int): Thickness of the object counting region.
            line_dist_thresh (int): Euclidean distance threshold for line counter.
            cls_txtdisplay_gap (int): Display gap between each class count.
        """

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts
        self.line_dist_thresh = line_dist_thresh
        self.counting_region = None
        self.region_color = count_reg_color
        self.region_thickness = region_thickness

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        self.names = classes_names  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.cls_txtdisplay_gap = cls_txtdisplay_gap
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.track_color = track_color

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Initialize counting region
        if len(self.reg_pts) == 2:
            print("Line Counter Initiated.")
            self.counting_region = LineString(self.reg_pts)
        elif len(self.reg_pts) >= 3:
            print("Polygon Counter Initiated.")
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        Handles mouse events for defining and moving the counting region in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters for the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    # def extract_and_process_tracks(self, tracks):
    #     """Extracts and processes tracks for object counting in a video stream."""
    #     self.items_status = []
    #     # Annotator Init and region drawing
    #     self.annotator = Annotator(self.im0, self.tf, self.names)

    #     # Draw region or line
    #     self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

    #     if tracks[0].boxes.id is not None:
    #         boxes = tracks[0].boxes.xyxy.cpu()
    #         clss = tracks[0].boxes.cls.cpu().tolist()
    #         track_ids = tracks[0].boxes.id.int().cpu().tolist()
            
    #         self.item_status= {}
    #         # Extract tracks
    #         for box, track_id, cls in zip(boxes, track_ids, clss):
    #             self.item_status["class_id"]= cls
    #             self.item_status["track_id"] = track_id
    #             # Draw bounding box
    #             self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

    #             # Store class info
    #             if self.names[cls] not in self.class_wise_count:
    #                 self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

    #             # Draw Tracks
    #             track_line = self.track_history[track_id]
    #             track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
    #             if len(track_line) > 30:
    #                 track_line.pop(0)

    #             # Draw track trails
    #             if self.draw_tracks:
    #                 self.annotator.draw_centroid_and_tracks(
    #                     track_line,
    #                     color=self.track_color if self.track_color else colors(int(track_id), True),
    #                     track_thickness=self.track_thickness,
    #                 )

    #             prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

    #             # Count objects in any polygon
    #             if len(self.reg_pts) >= 3:
    #                 is_inside = self.counting_region.contains(Point(track_line[-1]))
    #                 self.item_status["status"] = is_inside

    #                 if prev_position is not None and is_inside and track_id not in self.count_ids:
    #                     self.count_ids.append(track_id)

    #                     if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
    #                         self.in_counts += 1
    #                         self.class_wise_count[self.names[cls]]["IN"] += 1
    #                     else:
    #                         self.out_counts += 1
    #                         self.class_wise_count[self.names[cls]]["OUT"] += 1

    #             # Count objects using line
    #             elif len(self.reg_pts) == 2:
    #                 if prev_position is not None and track_id not in self.count_ids:
    #                     distance = Point(track_line[-1]).distance(self.counting_region)
    #                     if distance < self.line_dist_thresh and track_id not in self.count_ids:
    #                         self.count_ids.append(track_id)

    #                         if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
    #                             self.in_counts += 1
    #                             self.class_wise_count[self.names[cls]]["IN"] += 1
    #                         else:
    #                             self.out_counts += 1
    #                             self.class_wise_count[self.names[cls]]["OUT"] += 1
    #             self.items_status.append(self.item_status)
    #     labels_dict = {}

    #     for key, value in self.class_wise_count.items():
    #         if value["IN"] != 0 or value["OUT"] != 0:
    #             if not self.view_in_counts and not self.view_out_counts:
    #                 continue
    #             elif not self.view_in_counts:
    #                 labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
    #             elif not self.view_out_counts:
    #                 labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
    #             else:
    #                 labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

    #     if labels_dict:
    #         self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)

    #     return self.items_status
    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        self.items_status = []
        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)
        # Draw region or line
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                item_status = {"class_name": self.names[cls], "track_id": track_id}  # Reset for each track
                
                # Draw bounding box
                #NOTE: not to draw box
                # self.annotator.box_label(box, label=f"{self.names[cls]}", color=colors(int(track_id), True))

                # Initialize class info if not present
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon region
                is_inside = False
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    item_status["is_inside"] = is_inside  # Update the inside status
                    item_status["is_outside"] = not is_inside  # Update the outside status

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"] += 1

                # Count objects crossing a line
                elif len(self.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)
                        if distance < self.line_dist_thresh and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["OUT"] += 1
                else:
                    item_status["is_inside"] = is_inside  # Default if not in polygon or line-based counting

                # Append item_status for each track
                self.items_status.append(item_status)

        labels_dict = {}
        for key, value in self.class_wise_count.items():
            if value["IN"] != 0 or value["OUT"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not self.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"
        #NOTE:stopping display analyics
        if False:
            if labels_dict:
                self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)

        return self.items_status


    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            cv2.imshow(self.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.item_status = self.extract_and_process_tracks(tracks)  # draw region even if no objects

        if self.view_img:
            self.display_frames()
        return self.im0, self.item_status


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectCounter(classes_names)

