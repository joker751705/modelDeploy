from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io

from app.api.dependencies import get_yolov8, get_detr
from app.inference.detr_detector_onnx import DetrDetector
from app.inference.yolov8_detector_onnx import YoloV8Detector

# 创建一个路由实例
router = APIRouter()

@router.post(
    "/yolov8",
    summary="目标检测",
    description="上传一张图片，返回标注了检测框的图片。"
)
async def detect_image(
    file: UploadFile = File(...),
    detector: YoloV8Detector = Depends(get_yolov8)
):
    # 读取和解码图像的逻辑保持不变
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无效的图像文件。")

    # 调用服务层的业务逻辑
    annotated_image = detector.predict(image)

    # 编码和返回图像的逻辑保持不变
    success, encoded_image = cv2.imencode(".png", annotated_image)
    if not success:
        raise HTTPException(status_code=500, detail="图像编码失败。")

    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")

@router.post(
    "/detr",
    summary="目标检测",
    description="上传一张图片，返回标注了检测框的图片。"
)
async def detect_image(
    file: UploadFile = File(...),
    detector: DetrDetector = Depends(get_detr)
):
    # 读取和解码图像的逻辑保持不变
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无效的图像文件。")

    # 调用服务层的业务逻辑
    annotated_image = detector.predict(image)

    # 编码和返回图像的逻辑保持不变
    success, encoded_image = cv2.imencode(".png", annotated_image)
    if not success:
        raise HTTPException(status_code=500, detail="图像编码失败。")

    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")