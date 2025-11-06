package com.backend.service;

import com.rabbitmq.client.Channel;
import com.backend.config.RabbitMQConfig;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.util.Map;

@Slf4j
@Service
public class ResultConsumerService {
    private final SimpMessagingTemplate messagingTemplate;

    public ResultConsumerService(SimpMessagingTemplate messagingTemplate) {
        this.messagingTemplate = messagingTemplate;
    }

    @RabbitListener(queues = RabbitMQConfig.RESULT_QUEUE)
    public void receiveImageResult(Map<String, String> resultMessage, Channel channel, Message amqpMessage) {
        try {
            String taskId = resultMessage.get("taskId");
            log.info("success");
            // 构造 WebSocket 的目标地址
            String destination = "/topic/ai-results/" + taskId;

            // 通过 WebSocket 主动推送消息给订阅了该地址的客户端
            messagingTemplate.convertAndSend(destination, resultMessage);
            channel.basicAck(amqpMessage.getMessageProperties().getDeliveryTag(), false);
        }  catch (Exception e) {
            log.error(e.getMessage());
        }
    }
}
