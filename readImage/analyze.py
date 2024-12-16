# image = cv2.imread('./readImage/assets/2024-12-12_173716.png')
import cv2
import numpy as np

# 加载YOLO模型配置文件和权重文件
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# 加载COCO数据集标签（YOLOv3常用）
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 获取YOLO的输出层
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

# 加载图像
image = cv2.imread('path_to_your_image.jpg')
height, width, channels = image.shape

# 将图像转换为YOLO网络输入格式（归一化、尺寸调整）
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo_net.setInput(blob)

# 执行前向传播以获取检测结果
outs = yolo_net.forward(output_layers)

# 初始化用于存储检测结果的变量
class_ids = []
confidences = []
boxes = []

# 处理检测结果
for out in outs:
    for detection in out:
        scores = detection[5:]  # 类别得分
        class_id = np.argmax(scores)  # 得分最高的类别索引
        confidence = scores[class_id]  # 类别的置信度
        
        # 过滤低置信度的检测结果
        if confidence > 0.5:
            # 获取边界框的坐标
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # 计算矩形框的左上角坐标
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            # 存储检测结果
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 使用非极大值抑制来减少冗余的框
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 统计检测到的物体数量
object_count = len(indices) if indices is not None else 0

# 在图像上绘制检测结果并显示
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    
    # 绘制边界框和标签
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 在左上角显示检测到的物体数量
cv2.putText(image, f'Objects detected: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 显示图像
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()