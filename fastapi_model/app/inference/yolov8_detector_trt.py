import cv2
import numpy as np

from app.inference.base.tensorrt_base import TensorRTBase


class YoloV8DetectorTRT(TensorRTBase):

    def __init__(self, model_path, input_shape=(640, 640),
                 confidence_thres=0.5, iou_thres=0.5,
                 engine_has_nms=False):
        super().__init__(model_path)
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.input_height, self.input_width = input_shape
        self.engine_has_nms = engine_has_nms

        # 按需修改类别名
        self.class_names = self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def _preprocess(self, image, context):
        """
        图像预处理：letterbox缩放、归一化、维度转换（与ONNX版一致）
        """
        context["original_image"] = image
        self.img_height, self.img_width = image.shape[:2]

        ratio = min(self.input_width / self.img_width, self.input_height / self.img_height)
        new_width = int(self.img_width * ratio)
        new_height = int(self.img_height * ratio)
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        start_x = (self.input_width - new_width) // 2
        start_y = (self.input_height - new_height) // 2
        canvas[start_y:start_y + new_height, start_x:start_x + new_width, :] = resized_img

        input_tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1,3,H,W)
        return {self.input_names[0]: np.ascontiguousarray(input_tensor)}

    def _postprocess(self, outputs, context):
        out = outputs[self.output_names[0]]
        # 规范到 (N, num_attrs)
        if out.ndim == 3 and out.shape[1] < out.shape[2]:
            predictions = np.squeeze(out, axis=0).T
        elif out.ndim == 3:
            predictions = np.squeeze(out, axis=0)
        else:
            predictions = out

        if predictions.shape[0] == 0:
            return context["original_image"]

        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.confidence_thres
        predictions = predictions[mask]
        scores = scores[mask]

        if predictions.shape[0] == 0:
            return context["original_image"]  # ← 增加这一行以防筛完空

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes_xyxy = self.extract_boxes(predictions)

        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        boxes_xywh_for_nms = np.column_stack((x1, y1, x2 - x1, y2 - y1)).astype(np.int32)

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh_for_nms.tolist(), scores.tolist(),
            self.confidence_thres, self.iou_thres
        )
        # NMS 之后为空也返回原图
        if isinstance(indices, tuple) or len(indices) == 0:
            return context["original_image"]  # ← 改这里

        indices = np.array(indices).reshape(-1)
        final_boxes = boxes_xyxy[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]
        result = self.draw_detections(context["original_image"], final_boxes, final_scores, final_class_ids)
        return result

    def extract_boxes(self, predictions):
        """
        从预测中提取边界框并转换回原始图像坐标（cx,cy,w,h -> xyxy）
        """
        boxes = predictions[:, :4].copy()

        ratio = min(self.input_width / self.img_width, self.input_height / self.img_height)
        pad_x = (self.input_width - self.img_width * ratio) / 2
        pad_y = (self.input_height - self.img_height * ratio) / 2

        # 去 padding，缩放回原图尺度
        boxes[:, 0] = (boxes[:, 0] - pad_x) / ratio  # cx
        boxes[:, 1] = (boxes[:, 1] - pad_y) / ratio  # cy
        boxes[:, 2] /= ratio  # w
        boxes[:, 3] /= ratio  # h

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        xyxy = np.column_stack((x1, y1, x2, y2))
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, self.img_width - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, self.img_height - 1)
        return xyxy.astype(np.int32)

    def draw_detections(self, image, boxes, scores, class_ids):
        """
        在图像上绘制检测结果
        """
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{self.class_names[class_id]}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image