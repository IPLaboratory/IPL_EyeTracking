import torch
import cv2
import pathlib
import numpy as np

# Adjust pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the 'speaker' detection model
model_light = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='ML/Models/light_best_yolo5.pt', force_reload=True
)
model_light.conf = 0.6  # Confidence threshold for 'speaker'
model_light.iou = 0.6   # IoU threshold for 'speaker'

# Detection settings for 'speaker'
min_width_light = 5
min_height_light = 5
max_width_light = 5000
max_height_light = 5000
min_aspect_ratio_light = 1.2  # 가로가 세로보다 1.2배 이상 긴 경우

# Load the 'fan' detection model
model_fan = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='ML/Models/best_fan_yolo5t.pt', force_reload=True
)
model_fan.conf = 0.88  # Confidence threshold for 'fan'
model_fan.iou = 0.88   # IoU threshold for 'fan'

# Detection settings for 'fan'
min_width_fan = 50
min_height_fan = 50
max_width_fan = 500
max_height_fan = 500
min_aspect_ratio_fan = 1.2

# 갈색 필터 설정 (대략적인 RGB 범위)
brown_lower = np.array([50, 30, 0], dtype=np.uint8)
brown_upper = np.array([160, 110, 60], dtype=np.uint8)

# 바운딩 박스를 확장하는 함수
def expand_box(x1, y1, x2, y2, scale=0.1):
    width = x2 - x1
    height = y2 - y1
    # 확장된 크기 계산
    new_x1 = x1 - width * scale / 2
    new_y1 = y1 - height * scale / 2
    new_x2 = x2 + width * scale / 2
    new_y2 = y2 + height * scale / 2
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the highest confidence object for 'light'
    results_light = model_light(frame)
    max_conf_light = None
    best_det_light = None
    for det in results_light.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = model_light.names[int(cls)]
        if label == 'light':
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / height  # 가로가 세로보다 1.2배 이상 긴지 확인
            if (
                min_width_light <= width <= max_width_light and
                min_height_light <= height <= max_height_light and
                aspect_ratio >= min_aspect_ratio_light
            ):
                # 테두리의 갈색을 확인하기 위해 바운딩 박스 주변 픽셀 추출
                border_pixels = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # 갈색 검출
                brown_mask = cv2.inRange(border_pixels, brown_lower, brown_upper)
                brown_ratio = np.sum(brown_mask) / (width * height)

                # 갈색 비율이 충분할 때만 인식
                if brown_ratio > 0.03 and (max_conf_light is None or conf > max_conf_light):
                    max_conf_light = conf
                    best_det_light = (x1, y1, x2, y2, conf, label)

    # Draw the best detection for 'light' with expanded bounding box
    if best_det_light:
        x1, y1, x2, y2, conf, label = best_det_light
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, scale=0.1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    # Detect the highest confidence object for 'fan'
    results_fan = model_fan(frame)
    max_conf_fan = None
    best_det_fan = None
    for det in results_fan.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = model_fan.names[int(cls)]
        if label == 'fan':
            width, height = x2 - x1, y2 - y1
            aspect_ratio = height / width
            if (
                min_width_fan <= width <= max_width_fan and
                min_height_fan <= height <= max_height_fan and
                aspect_ratio >= min_aspect_ratio_fan
            ):
                if max_conf_fan is None or conf > max_conf_fan:
                    max_conf_fan = conf
                    best_det_fan = (x1, y1, x2, y2, conf, label)

    # Draw the best detection for 'fan' with expanded bounding box
    if best_det_fan:
        x1, y1, x2, y2, conf, label = best_det_fan
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, scale=0.1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Display the resulting frame
    cv2.imshow('Webcam Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
