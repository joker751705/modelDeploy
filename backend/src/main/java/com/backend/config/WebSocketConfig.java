package com.backend.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // 启用一个简单的内存消息代理，并为 /topic 前缀的目标启用它
        config.enableSimpleBroker("/topic");
        // 设置应用目标前缀，客户端发送消息到 /app/...
        config.setApplicationDestinationPrefixes("/app");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // 注册 /ws-connect 作为 STOMP 的端点，并允许 SockJS 回退选项
        registry.addEndpoint("/ws-connect").setAllowedOrigins("http://127.0.0.1:3000", "http://localhost:3000").withSockJS();
    }
}