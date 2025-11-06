package com.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient.Builder webClient() {
        final int maxInMemorySize = 64 * 1024 * 1024;

        // 2. 创建一个交换策略 (ExchangeStrategies) 并应用新的缓冲区大小
        final ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(codecs -> codecs.defaultCodecs().maxInMemorySize(maxInMemorySize))
                .build();

        // 3. 在创建WebClient实例时，使用这个自定义的交换策略
        return WebClient.builder()
                .exchangeStrategies(strategies);
    }
}