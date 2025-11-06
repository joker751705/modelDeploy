package com.backend.config;

import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.config.SimpleRabbitListenerContainerFactory;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitMQConfig {

    public static final String TASK_EXCHANGE = "ai.task.exchange";
    public static final String TASK_QUEUE = "ai.task.queue";
    public static final String TASK_ROUTING_KEY = "ai.task.routing";

    public static final String RESULT_EXCHANGE = "ai.result.exchange";
    public static final String RESULT_QUEUE = "ai.result.queue";
    public static final String RESULT_ROUTING_KEY = "ai.result.routing";

    @Bean
    public Jackson2JsonMessageConverter messageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    //用于发送消息
    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);
        //设置JSON转换器
        template.setMessageConverter(messageConverter());
        //如果没有找到匹配的队列，将这条消息返回给生产者
        template.setMandatory(true);
        return template;
    }

    //配置所有消费者的行为
    @Bean
    public SimpleRabbitListenerContainerFactory rabbitListenerContainerFactory(ConnectionFactory connectionFactory) {
        SimpleRabbitListenerContainerFactory factory = new SimpleRabbitListenerContainerFactory();
        factory.setConnectionFactory(connectionFactory);
        //设置收到的消息为JSON格式
        factory.setMessageConverter(messageConverter());
        //设置手动确认模式
        factory.setAcknowledgeMode(AcknowledgeMode.MANUAL);
        //设置默认同时有3个线程处理消息
        factory.setConcurrentConsumers(3);
        //设置最大并发消费者的数量为10
        factory.setMaxConcurrentConsumers(10);
        return factory;
    }

    // 订单相关配置
    // 设置交换机名称，持久化，不自动删除，作为生产者的目标
    @Bean
    public DirectExchange taskExchange() {
        return new DirectExchange(TASK_EXCHANGE, true, false);
    }

    @Bean
    public Queue taskQueue() {
        return QueueBuilder.durable(TASK_QUEUE)
                .withArgument("x-message-ttl", 300000) // 5分钟TTL
                .build();
    }

    @Bean
    public Binding taskBinding() {
        return BindingBuilder.bind(taskQueue())
                .to(taskExchange())
                .with(TASK_ROUTING_KEY);
    }

    @Bean
    public DirectExchange resultExchange() {
        return new DirectExchange(RESULT_EXCHANGE, true, false);
    }

    @Bean
    public Queue resultQueue() {
        return QueueBuilder.durable(RESULT_QUEUE)
                .withArgument("x-message-ttl", 300000) // 5分钟TTL
                .build();
    }

    @Bean
    public Binding resultBinding() {
        return BindingBuilder.bind(resultQueue())
                .to(resultExchange())
                .with(RESULT_ROUTING_KEY);
    }
}