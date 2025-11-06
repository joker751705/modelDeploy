import base64
import json

import cv2
import numpy as np
import pika

from app.core.config import settings


def image_data_processor(body, model):
    """消息回调函数"""
    task_message = json.loads(body)
    task_id = task_message.get('taskId')
    base64_image_data = task_message.get('imageData')

    print(f"\n [x] 收到图像处理任务, ID: {task_id}")

    #  对图像数据进行 Base64 解码，得到原始的字节数组
    image_bytes = base64.b64decode(base64_image_data)

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    annotated_image = model.predict(image)
    success, buffer = cv2.imencode('.jpg', annotated_image)

    if not success:
        raise ValueError("无法将处理后的图像编码为 JPEG 格式")

    base64_result_image = base64.b64encode(buffer.tobytes()).decode('utf-8')

    # 准备并发送结果消息
    result_message = {'taskId': task_id, 'result': base64_result_image, 'imageFormat': 'image/jpeg', 'status': 'success'}

    return result_message
