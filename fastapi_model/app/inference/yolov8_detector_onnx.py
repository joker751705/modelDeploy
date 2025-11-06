import cv2
import numpy as np

from app.inference.base.onnx_base import OnnxBase


class YoloV8Detector(OnnxBase):
    def __init__(self, model_path, input_shape=(640, 640), confidence_thres=0.5, iou_thres=0.5):
        super().__init__(model_path)
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.input_height, self.input_width = input_shape
        self.class_names = self.class_names = [
          'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
          'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
          'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
          'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
          'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
          'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
          'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
          'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
          'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
          'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
        ]

    def _preprocess(self, image, context):
        """
        图像预处理：letterbox缩放、归一化、维度转换
        """
        context["original_image"] = image
        self.img_height, self.img_width = image.shape[:2]

        # 1. Letterbox缩放
        # 计算缩放比例，保持长宽比
        ratio = min(self.input_width / self.img_width, self.input_height / self.img_height)
        new_width = int(self.img_width * ratio)
        new_height = int(self.img_height * ratio)
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 创建一个灰色画布，并将缩放后的图像粘贴到中央
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)

        # 计算粘贴位置
        start_x = (self.input_width - new_width) // 2
        start_y = (self.input_height - new_height) // 2
        canvas[start_y:start_y + new_height, start_x:start_x + new_width, :] = resized_img

        # 2. 转换数据格式
        # BGR -> RGB, HWC -> CHW, 归一化
        input_tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)  # 增加batch维度

        return {self.input_names[0]: input_tensor}

    def _postprocess(self, model_outputs, context):
        """
        模型输出后处理：解码、过滤、NMS
        """
        predictions = np.squeeze(model_outputs[0]).T  # (1, num_attrs, 8400) -> (8400, num_attrs)

        # 过滤掉置信度低的预测
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres]

        if predictions.shape[0] == 0:
            return context["original_image"]

        # 获取类别ID
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # 将 cx,cy,w,h 格式的 boxes 转换为 x1,y1,x2,y2 格式
        boxes = self.extract_boxes(predictions)

        # 应用非极大值抑制 (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]
        result_image = self.draw_detections(context["original_image"], final_boxes, final_scores, final_class_ids)

        return result_image

    def extract_boxes(self, predictions):
        """
        从预测中提取边界框并转换回原始图像坐标
        """
        # 提取box部分
        boxes = predictions[:, :4]

        # 尺寸恢复 (从640x640缩放回原始图像尺寸)
        # 1. 计算缩放比例和填充
        ratio = min(self.input_width / self.img_width, self.input_height / self.img_height)
        pad_x = (self.input_width - self.img_width * ratio) / 2
        pad_y = (self.input_height - self.img_height * ratio) / 2

        # 2. 坐标转换
        boxes[:, 0] = (boxes[:, 0] - pad_x) / ratio  # x_center
        boxes[:, 1] = (boxes[:, 1] - pad_y) / ratio  # y_center
        boxes[:, 2] /= ratio  # width
        boxes[:, 3] /= ratio  # height

        # 3. cx,cy,w,h -> x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        return np.column_stack((x1, y1, x2, y2)).astype(np.int32)

    def draw_detections(self, image, boxes, scores, class_ids):
        """
        在图像上绘制检测结果
        """
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box

            # 绘制框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 准备标签文字
            label = f"{self.class_names[class_id]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # 绘制标签背景
            cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)

            # 绘制标签文字
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image