package com.backend.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * 全局 Web 配置类
 * 这里专门配置与 Spring MVC 相关的全局规则，例如 CORS。
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    /**
     * 配置全局跨域资源共享 (CORS) 规则。
     * 这个配置会为所有 HTTP 端点（包括 SockJS 的握手请求）添加必要的 CORS 响应头。
     */
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**") // 匹配应用中的所有 URL
                .allowedOrigins("http://127.0.0.1:3000", "http://localhost:3000") // 允许来自您前端应用的源
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS") // 允许的 HTTP 方法
                .allowedHeaders("*") // 允许所有的请求头
                .allowCredentials(true); // 允许客户端发送凭证信息（如 Cookie）
    }
}
