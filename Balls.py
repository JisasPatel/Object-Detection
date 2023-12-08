#!/Users/jisaspatel/opt/anaconda3/bin/python
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

print("ok")



ZONE_POLYGON_1 = np.array([
    [1250,550],
   [1780,550],
   [1780,1000],
   [1250,1000]
])

ZONE_POLYGON_2 = np.array([
   [780,550],
   [1225,550],
   [1225,1000],
   [780,1000]
])

ZONE_POLYGON_3 = np.array([
   [780,25],
   [1225,25],
   [1225,510],
   [780,510]
])


ZONE_POLYGON_4 = np.array([
   [1250,25],
   [1780,25],
   [1780,510],
   [1250,510]
])





def main():
  cap = cv2.VideoCapture("/Users/jisaspatel/Desktop/YOLO-Pic/video.mp4")
  model = YOLO("best.pt")

  box_annotator = sv.BoxAnnotator(
     thickness = 2,
     text_thickness=2,
     text_scale = 1
  )
  colors = sv.ColorPalette.default()
  zone1 = sv.PolygonZone(polygon=ZONE_POLYGON_1,frame_resolution_wh = (1920,1080))
  zone2 = sv.PolygonZone(polygon=ZONE_POLYGON_2,frame_resolution_wh = (1920,1080))
  zone3 = sv.PolygonZone(polygon=ZONE_POLYGON_3,frame_resolution_wh = (1920,1080))
  zone4 = sv.PolygonZone(polygon=ZONE_POLYGON_4,frame_resolution_wh = (1920,1080))

  zone_annotator_1 = sv.PolygonZoneAnnotator( 
                      zone=zone1, 
                      color=colors.by_idx(0), 
                      thickness = 2,
                      text_thickness =4,
                      text_scale = 2 
    )
  zone_annotator_2 = sv.PolygonZoneAnnotator( 
                      zone=zone2, 
                      color=colors.by_idx(0), 
                      thickness = 2,
                      text_thickness =4,
                      text_scale = 2 )


  zone_annotator_3 = sv.PolygonZoneAnnotator( 
                      zone=zone3, 
                      color=colors.by_idx(0), 
                      thickness = 2,
                      text_thickness =4,
                      text_scale = 2 
    )
  zone_annotator_4 = sv.PolygonZoneAnnotator( 
                      zone=zone4, 
                      color=colors.by_idx(0), 
                      thickness = 2,
                      text_thickness =4,
                      text_scale = 2 )




  while True:
    ret, frame = cap.read()

    # print(frame.shape)
    # break
    if not ret or frame is None:
        print("Error: Unable to capture frame")
        break
    results = model(frame,agnostic_nms = True)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id != 4]
    
    print("THIS IS ----",detections)

    bounding_boxes = detections.xyxy
    confidences = detections.confidence
    class_ids = detections.class_id


    labels = [ f"{model.model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(class_ids, confidences)]

    frame = box_annotator.annotate(scene=frame, detections = detections, labels= labels)

    zone1.trigger(detections= detections)
    zone2.trigger(detections= detections)
    zone3.trigger(detections= detections)
    zone4.trigger(detections= detections)
    
    frame = zone_annotator_1.annotate(scene=frame)
    frame = zone_annotator_2.annotate(scene=frame)
    frame = zone_annotator_3.annotate(scene=frame)
    frame = zone_annotator_4.annotate(scene=frame)


    cv2.imshow("yolov8",frame)

    if (cv2.waitKey(30) == 27):
      break
main()