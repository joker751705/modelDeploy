package com.backend.service;

import com.backend.config.AiServiceProperties;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.IOException;

@Service
public class AIService {

    private final WebClient webClient;
    private final AiServiceProperties properties;

    // 构造函数注入 WebClient 和配置属性类
    public AIService(WebClient.Builder webClientBuilder, AiServiceProperties properties) {
        // 最佳实践：为WebClient设置基础URL
        this.webClient = webClientBuilder.baseUrl(properties.getUrl()).build();
        this.properties = properties;
    }

    /**
     * 将图片文件代理到 FastAPI 服务 (非阻塞式)
     * @param imageFile 从前端接收到的图片文件
     * @param model 要使用的模型名称
     * @return 一个包含 FastAPI 返回的二进制图片数据的 Mono
     */
    public Mono<byte[]> processImage(MultipartFile imageFile, String model) {

        // 1. 从配置中安全地获取目标端点路径
        String endpoint = properties.getEndpoints().get(model);
        if (endpoint == null) {
            // 抛出具体的异常
            return Mono.error(new IllegalArgumentException("Unsupported AI model: " + model));
        }

        // 2. 将 MultipartFile 转换为响应式友好的数据流
        //    同时处理潜在的 IOException
        return Mono.fromCallable(imageFile::getBytes)
                .map(bytes -> new ByteArrayResource(bytes) {
                    @Override
                    public String getFilename() {
                        return imageFile.getOriginalFilename();
                    }
                })
                .flatMap(resource -> {
                    // 3. 构建 multipart/form-data 请求体
                    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                    body.add("file", resource);

                    // 4. 发起异步 POST 请求，并直接返回 Mono
                    return webClient.post()
                            .uri(endpoint) // 使用从配置中获取的路径
                            .contentType(MediaType.MULTIPART_FORM_DATA)
                            .body(BodyInserters.fromMultipartData(body))
                            .retrieve()
                            .bodyToMono(byte[].class);
                })
                .doOnError(IOException.class, e -> {
                    // 将受检异常包装为非受检异常
                    throw new RuntimeException("Failed to read image file bytes", e);
                });
    }
}