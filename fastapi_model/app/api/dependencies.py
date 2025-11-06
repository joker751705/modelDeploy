from fastapi import HTTPException

# 这个字典将在 main.py 的 lifespan 中被填充，用于在应用各部分间共享资源
LIFESPAN_CONTEXT = {}

def get_yolov8():
    """FastAPI的依赖项，返回全局唯一的detector实例。"""
    detector = LIFESPAN_CONTEXT.get("yolov8_instance")
    if detector is None:
        raise HTTPException(status_code=503, detail="模型服务当前不可用，请稍后重试。")
    return detector

def get_yolov8_trt():
    detector = LIFESPAN_CONTEXT.get("yolov8_trt_instance")
    if detector is None:
        raise HTTPException(status_code=503, detail="模型服务当前不可用，请稍后重试。")
    return detector

def get_detr():
    detector = LIFESPAN_CONTEXT.get("detr_instance")
    if detector is None:
        raise HTTPException(status_code=503, detail="模型服务当前不可用，请稍后重试。")
    return detector

def get_maskdino():
    model = LIFESPAN_CONTEXT.get("maskdino_instance")
    if model is None:
        raise HTTPException(status_code=503, detail="模型服务当前不可用，请稍后重试。")
    return model