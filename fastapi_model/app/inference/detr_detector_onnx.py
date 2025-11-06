import cv2
import numpy as np
from PIL import Image  # 推荐使用 PIL 进行图像尺寸变换

from app.inference.base.onnx_base import OnnxBase


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    """
    对 NumPy 数组在最后一个轴上计算 softmax，并保证数值稳定性。
    """
    # 减去最大值以防止指数爆炸
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def box_cxcywh_to_xyxy_numpy(x: np.ndarray) -> np.ndarray:
    """
    将 (center_x, center_y, width, height) 格式的边界框转换为 (x1, y1, x2, y2) 格式。
    """
    # 使用 NumPy 切片代替 torch.unbind
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [
        (x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)
    ]
    # 使用 np.stack 代替 torch.stack
    return np.stack(b, axis=-1)


# resize 函数保持不变，因为它操作的是 PIL Image，不依赖 PyTorch
def resize_image(image: Image.Image, size: int, max_size: int = 1333) -> Image.Image:
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            # PIL resize expects (width, height)
            return size
        else:
            # get_size_with_aspect_ratio returns (height, width)
            h, w = get_size_with_aspect_ratio(image_size, size, max_size)
            return (w, h)

    # F.resize is just a wrapper around PIL's resize
    new_size = get_size(image.size, size, max_size)
    rescaled_image = image.resize(new_size, Image.BILINEAR)
    return rescaled_image


class DetrDetector(OnnxBase):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # 类别名称保持不变
        self.class_names = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, data: np.ndarray, context: dict) -> dict:
        # data 是一个 BGR 格式的 NumPy 数组
        context["original_image"] = data.copy()  # 保存副本以防被修改

        # 1. BGR -> RGB -> PIL Image
        image_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # 2. 保存原始尺寸
        original_w, original_h = image_pil.size
        context["original_size"] = (original_w, original_h)

        # 3. Resize 图像
        rescaled_image_pil = resize_image(image_pil, size=800, max_size=1333)
        rescaled_w, rescaled_h = rescaled_image_pil.size
        context["rescaled_size"] = (rescaled_w, rescaled_h)

        # 4. PIL Image -> NumPy, 并执行 F.to_tensor 和 F.normalize 的逻辑
        # a. 转为 NumPy 数组并缩放到 [0, 1]
        image_np = np.array(rescaled_image_pil, dtype=np.float32) / 255.0

        # b. 从 (H, W, C) 转为 (C, H, W)
        image_np = image_np.transpose((2, 0, 1))

        # c. 标准化
        # Reshape mean and std to (3, 1, 1) for broadcasting
        image_np = (image_np - self.mean[:, None, None]) / self.std[:, None, None]

        # 5. 增加 batch 维度
        image_np = np.expand_dims(image_np, axis=0)

        # 假设 input_names[0] 是你模型输入的名称
        return {self.input_names[0]: image_np}

    def _postprocess(self, model_outputs: list, context: dict) -> np.ndarray:
        # model_outputs 是 onnxruntime 返回的 NumPy 数组列表
        # 1. 获取 ONNX 输出
        out_logits = model_outputs[0]
        out_bbox = model_outputs[1]

        # 2. 从 Logits 获取分数和标签
        prob = softmax_numpy(out_logits)
        # 在最后一个维度上获取最大值（分数）和其索引（标签）
        scores = np.max(prob[..., :-1], axis=-1)
        labels = np.argmax(prob[..., :-1], axis=-1)

        # 3. 根据阈值筛选
        mask = scores > 0.7
        scores = scores[mask]
        labels = labels[mask]
        out_bbox = out_bbox[mask]

        # 4. 坐标转换
        # a. 将中心点格式转为角点格式
        boxes = box_cxcywh_to_xyxy_numpy(out_bbox)

        # b. 将相对坐标 [0, 1] 缩放到 resized 图像的绝对坐标
        rescaled_w, rescaled_h = context["rescaled_size"]
        scale_fct = np.array([rescaled_w, rescaled_h, rescaled_w, rescaled_h], dtype=np.float32)
        boxes = boxes * scale_fct

        # c. 将坐标从 resized 图像映射回原始图像
        original_w, original_h = context["original_size"]
        w_ratio = original_w / rescaled_w
        h_ratio = original_h / rescaled_h
        boxes[:, [0, 2]] *= w_ratio
        boxes[:, [1, 3]] *= h_ratio

        # boxes, scores, labels 已经是 NumPy 数组了
        return self.draw_detections(context["original_image"], boxes, scores, labels)

    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                        class_ids: np.ndarray) -> np.ndarray:
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{self.class_names[class_id]}: {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image