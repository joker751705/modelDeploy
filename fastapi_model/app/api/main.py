# main.py
import threading
from contextlib import asynccontextmanager

import pika
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.api.routers import detection
from app.inference.detr_detector_onnx import DetrDetector
from app.inference.yolov8_detector_onnx import YoloV8Detector
from app.api.dependencies import LIFESPAN_CONTEXT
import io

from starlette.middleware.cors import CORSMiddleware

from app.inference.yolov8_detector_trt import YoloV8DetectorTRT
from app.workers.consumer import RabbitMQConsumer
from app.workers.processor import image_data_processor


# --- 1. 配置与模型加载 ---

# 创建FastAPI应用

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器"""
    # --- 启动事件 ---
    print("--- FastAPI 应用启动中 ---")

    # 1. 加载模型并存入上下文
    print("步骤 1: 加载模型...")
    yolov8 = YoloV8Detector(model_path=settings.YOLOv8_MODEL_PATH)
    LIFESPAN_CONTEXT["yolov8_instance"] = yolov8

    yolov8_trt = YoloV8DetectorTRT(model_path=settings.YOLOv8_TRT_MODEL_PATH)
    LIFESPAN_CONTEXT["yolov8_trt_instance"] = yolov8_trt

    detr = DetrDetector(model_path=settings.DETR_MODEL_PATH)
    LIFESPAN_CONTEXT["detr_instance"] = detr

    # 2. 连接 RabbitMQ
    print("步骤 2: 连接到 RabbitMQ...")
    credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASS)
    connection_params = pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        virtual_host=settings.RABBITMQ_VHOST,
        credentials=credentials,
        heartbeat=600  # 推荐设置心跳，防止连接因网络不活动而被断开
    )
    connection = pika.BlockingConnection(connection_params)
    LIFESPAN_CONTEXT["rmq_connection"] = connection

    # 3. 启动消费者线程
    consumer_instance = RabbitMQConsumer(
        connection=connection,
        model=yolov8_trt,
        processing_function=image_data_processor,
        task_queue=settings.TASK_QUEUE,
        result_exchange=settings.RESULT_EXCHANGE,
        result_routing_key=settings.RESULT_ROUTING_KEY
    )
    print("步骤 3: 启动 RabbitMQ 消费者线程...")
    consumer = threading.Thread(
        target=consumer_instance.run,
        daemon=True
    )
    LIFESPAN_CONTEXT["consumer_thread"] = consumer
    consumer.start()

    print("--- 应用启动完成 ---")

    yield  # 应用在此运行

    # --- 关闭事件 ---
    print("\n--- FastAPI 应用关闭中 ---")
    conn = LIFESPAN_CONTEXT.get("rmq_connection")
    if conn and conn.is_open:
        print("正在关闭 RabbitMQ 连接...")
        conn.close()

    thread = LIFESPAN_CONTEXT.get("consumer_thread")
    if thread and thread.is_alive():
        thread.join(timeout=5)

    print("--- 应用已关闭 ---")
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # 允许访问的源
    allow_credentials=True, # 支持 cookie
    allow_methods=["*"],    # 允许所有方法
    allow_headers=["*"],    # 允许所有请求头
)
app.include_router(detection.router, prefix="/api/detection", tags=["Detection"])

@app.get("/", summary="根路径")
def root():
    return {"message": "model"}