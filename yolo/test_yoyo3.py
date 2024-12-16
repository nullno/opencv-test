import cv2
import numpy as np

# 加载 YOLO 模型
model_cfg = "./yolo/v3/yolov3.cfg"  # 配置文件
model_weights = "./yolo/v3/yolov3.weights"  # 预训练权重
coco_names = "./yolo/v3/coco.names"  # COCO 类别文件

# 读取类别名称
with open(coco_names, "r") as f:
    classes = f.read().strip().split("\n")

# 加载网络
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 定义需要检测的类别
target_classes = {"person", "car"}  # 只检测人和车

# 获取输出层名称
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 处理图像
def detect_objects(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # 遍历每个输出层
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 检测置信度阈值
            if confidence > 0.5 and classes[class_id] in target_classes:
                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0) if classes[class_ids[i]] == "person" else (255, 0, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# 测试视频或图像
cap = cv2.VideoCapture("./yolo/assets/8698-213454544_medium.mp4")  # 读取摄像头，可以改为视频文件路径
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = detect_objects(frame)
    cv2.imshow("YOLO Detection", result_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
