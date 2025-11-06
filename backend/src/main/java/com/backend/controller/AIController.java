package com.backend.controller;

import com.backend.service.AIService;
import com.backend.service.TaskProducerService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;

import java.util.Map;

@RestController
@RequestMapping("/api/ai")
public class AIController {

    private final AIService aiService;
    private final TaskProducerService taskProducerService;

    public AIController(AIService aiService, TaskProducerService taskProducerService) {
        this.aiService = aiService;
        this.taskProducerService = taskProducerService;
    }

    @PostMapping("/image/{model}")
    public Mono<ResponseEntity<byte[]>> image(
            @PathVariable String model,
            @RequestParam("image") MultipartFile imageFile) {
        return aiService.processImage(imageFile, model)
                .map(imageBytes -> ResponseEntity.ok()
                        .contentType(MediaType.IMAGE_JPEG)
                        .body(imageBytes))
                .defaultIfEmpty(ResponseEntity.notFound().build());
    }

    @PostMapping(path = "/process-image/{model}")
    public ResponseEntity<Map<String, String>> submitImageTask(
            @PathVariable String model,
            @RequestParam("image") MultipartFile image) {
        if (image.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        String taskId = taskProducerService.sendImageAITask(image);
        return ResponseEntity.accepted().body(Map.of("taskId", taskId));
    }
}
