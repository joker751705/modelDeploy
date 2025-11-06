package com.backend.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.Map;

@ConfigurationProperties(prefix = "ai.fastapi.service")
@Data
public class AiServiceProperties {
    private String url;
    private Map<String, String> endpoints;
}
