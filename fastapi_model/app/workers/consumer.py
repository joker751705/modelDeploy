import json
import os
from functools import partial
from typing import Callable

import pika

from app.core.config import settings
from pika.exceptions import ConnectionClosedByBroker

class RabbitMQConsumer:
    def __init__(self, connection, model, processing_function: Callable, task_queue, result_exchange, result_routing_key):
        self.connection = connection
        self.model = model
        self.processing_function = processing_function
        self.task_queue = task_queue
        self.result_exchange = result_exchange
        self.result_routing_key = result_routing_key
        self.task_channel = self.connection.channel()
        self.result_channel = self.connection.channel()
        self._setup_channels()

    def _setup_channels(self):
        self.task_channel.basic_qos(prefetch_count=1)

    def _on_message(self, ch, method, properties, body):
        print(f"\n [x] 收到图像处理任务: {self.task_queue}")
        try:
            result_message = self.processing_function(body, self.model)
            self.result_channel.basic_publish(
                exchange=self.result_exchange,
                routing_key=self.result_routing_key,
                body=json.dumps(result_message),
                properties=pika.BasicProperties(content_type='application/json')
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f" [!] 任务处理失败: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


    def run(self):
        """启动消费循环。这个方法会阻塞，应该在独立线程中运行。"""
        self.task_channel.basic_consume(
            queue=self.task_queue,
            on_message_callback=self._on_message
        )
        print(' [*] 等待任务中... (按 CTRL+C 退出)')
        try:
            self.task_channel.start_consuming()
        except ConnectionClosedByBroker:
            print(" [i] RabbitMQ 连接已关闭，消费者线程正常退出。")
        except Exception as e:
            print(f" [!] 消费者线程异常退出: {e}")
        finally:
            print(" [*] 消费者线程已停止。")