import { Client } from '@stomp/stompjs';
import SockJS from 'sockjs-client';

class WebSocketService {
    constructor() {
        this.stompClient = null;
        this.connectionPromise = null;
        this.subscriptions = new Map();
    }

    connect() {
        if (!this.connectionPromise) {
            this.connectionPromise = new Promise((resolve, reject) => {
                if (this.stompClient && this.stompClient.connected) {
                    resolve();
                    return;
                }

                const socket = new SockJS('http://localhost:8080/ws-connect'); // 你的WebSocket端点
                this.stompClient = new Client({
                    webSocketFactory: () => socket,
                    debug: (str) => { console.log(new Date(), str); },
                    onConnect: () => {
                        console.log('WebSocket Connected!');
                        resolve();
                    },
                    onStompError: (frame) => {
                        console.error('Broker reported error: ' + frame.headers['message']);
                        console.error('Additional details: ' + frame.body);
                        this.connectionPromise = null; // 连接失败，允许重试
                        reject(frame);
                    },
                });
                this.stompClient.activate();
            });
        }
        return this.connectionPromise;
    }

    disconnect() {
        if (this.stompClient && this.stompClient.connected) {
            this.stompClient.deactivate();
            console.log('WebSocket Disconnected');
        }
        this.stompClient = null;
        this.connectionPromise = null;
        this.subscriptions.clear();
    }

    subscribe(destination, callback) {
        if (this.stompClient && this.stompClient.connected) {
            const subscription = this.stompClient.subscribe(destination, callback);
            this.subscriptions.set(destination, subscription);
        } else {
            console.error('Cannot subscribe, Stomp client is not connected.');
        }
    }

    unsubscribe(destination) {
        const subscription = this.subscriptions.get(destination);
        if (subscription) {
            subscription.unsubscribe();
            this.subscriptions.delete(destination);
            console.log(`Unsubscribed from ${destination}`);
        }
    }
}

export default new WebSocketService();