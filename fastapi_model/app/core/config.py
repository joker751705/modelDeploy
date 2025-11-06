import os

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # 应用配置
    APP_TITLE: str = "模型部署"
    APP_DESCRIPTION: str = "使用FastApi部署的模型"

    # CORS 配置
    CORS_ORIGINS: List[str] = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    # 模型路径配置
    YOLOv8_MODEL_PATH: str = "models/yolov8/best.onnx"
    YOLOv8_TRT_MODEL_PATH: str = "models/yolov8/best_fp32.engine"
    DETR_MODEL_PATH: str = "models/detr/detr_model.onnx"
    MASKDINO_MODEL_PATH: str = "models/detr/maskdino_sem.onnx"

    RABBITMQ_USER: str = ""
    RABBITMQ_PASS: str = ""
    RABBITMQ_HOST: str = ""
    RABBITMQ_PORT: int = 5672
    RABBITMQ_VHOST: str = os.getenv('RABBITMQ_VHOST', '/')

    TASK_QUEUE: str = 'ai.task.queue'
    RESULT_EXCHANGE: str = 'ai.result.exchange'
    RESULT_ROUTING_KEY: str = 'ai.result.routing'

    class Config:
        # 如果你使用 .env 文件，这行可以让Pydantic从中读取配置
        env_file = ".env"
        env_file_encoding = 'utf-8'


# 创建一个全局可用的配置实例
settings = Settings()