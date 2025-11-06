package com.backend.service;

import com.backend.config.RabbitMQConfig;
import org.springframework.amqp.AmqpException;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Service
public class TaskProducerService {
    private final RabbitTemplate rabbitTemplate;

    public TaskProducerService(RabbitTemplate rabbitTemplate) {
        this.rabbitTemplate = rabbitTemplate;
    }

    public String sendImageAITask(MultipartFile file) {
        try {
            String taskId = UUID.randomUUID().toString();
            byte[] imageBytes = file.getBytes();
            String base64ImageData = Base64.getEncoder().encodeToString(imageBytes);
            Map<String, Object> taskMessage = new HashMap<>();
            taskMessage.put("taskId", taskId);
            taskMessage.put("originalFileName", file.getOriginalFilename());
            taskMessage.put("contentType", file.getContentType());
            taskMessage.put("imageData", base64ImageData);
            rabbitTemplate.convertAndSend(RabbitMQConfig.TASK_EXCHANGE, RabbitMQConfig.TASK_ROUTING_KEY, taskMessage);
            return taskId;
        } catch (AmqpException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}
