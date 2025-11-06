<script setup>
import { onUnmounted, ref, computed } from 'vue';
// 导入我们模拟的API函数
// 在实际项目中，你会从 @/api/predict.js 导入
import { predictImageWithWebSocket, predictImageWithHttp } from "@/api/predict";
import websocketService from "@/services/websocketService";

// --- 1. 模型配置中心 ---
// 在这里定义所有可用的模型。未来新增模型只需在此处添加一项。
const models = ref([
  {
    id: 'yolov8_http',
    name: 'YOLOv8 目标检测 (HTTP)',
    type: 'http', // 处理类型
    model: 'yolov8',
    apiFn: predictImageWithHttp // 对应的API调用函数
  },
  {
    id: 'yolov8_websocket',
    name: 'YOLOv8 目标检测 (WebSocket)',
    type: 'websocket', // 处理类型
    model: 'yolov8',
    apiFn: predictImageWithWebSocket // 对应的API调用函数
  },
  {
    id: 'detr',
    name: 'DETR 目标检测 (HTTP)',
    type: 'http', // 处理类型
    model: 'detr',
    apiFn: predictImageWithHttp // 对应的API调用函数
  }
]);

// --- 2. 响应式状态定义 ---

// 存储用户选择的原始文件对象
const selectedFile = ref(null);
// 当前选中的模型ID，默认为第一个
const selectedModelId = ref(models.value[0].id);

// 用于在本地预览用户上传的图片
const originalImageUrl = ref('');
// 用于显示从服务器返回的、已标注的图片
const resultImageUrl = ref('');
// 控制加载动画或文字的显示
const isLoading = ref(false);
// 存储和显示任何可能发生的错误信息
const errorMsg = ref('');

// 计算属性，方便地获取当前选中模型的所有信息
const selectedModel = computed(() => models.value.find(m => m.id === selectedModelId.value));

// --- 3. 方法定义 ---

/**
 * 当用户通过文件输入框选择了文件后触发此函数
 */
const handleFileUpload = (event) => {
  const file = event.target.files[0];
  if (!file) {
    selectedFile.value = null;
    originalImageUrl.value = '';
    return;
  }
  selectedFile.value = file;
  resultImageUrl.value = '';
  errorMsg.value = '';
  originalImageUrl.value = URL.createObjectURL(file);
};

/**
 * 点击“开始预测”按钮后触发此函数 - 核心逻辑分发器
 */
const submitImage = async () => {
  if (!selectedFile.value) {
    alert("请先选择一张图片！");
    return;
  }

  isLoading.value = true;
  errorMsg.value = '';
  resultImageUrl.value = '';

  try {
    // --- 根据所选模型的类型，执行不同的逻辑 ---
    if (selectedModel.value.type === 'http') {
      await handleHttpPrediction();
    } else if (selectedModel.value.type === 'websocket') {
      await handleWebSocketPrediction();
    }
  } catch (error) {
    errorMsg.value = "预测失败，请检查服务器或查看控制台获取更多信息。";
    console.error("Prediction failed:", error);
  } finally {
    isLoading.value = false;
  }
};

/**
 * 处理直接返回图像的HTTP请求
 */
const handleHttpPrediction = async () => {
  console.log(`正在使用HTTP模式为模型 [${selectedModel.value.name}] 进行预测...`);
  // 调用模型对应的API函数，它会直接返回一个Blob对象
  const imageBlob = await selectedModel.value.apiFn(selectedFile.value, selectedModel.value.model);

  // 将返回的Blob数据转换为可显示的URL
  resultImageUrl.value = URL.createObjectURL(imageBlob);
  console.log("HTTP模式预测成功，结果已显示。");
};

/**
 * 处理通过WebSocket返回结果的请求
 */
const handleWebSocketPrediction = async () => {
  console.log(`正在使用WebSocket模式为模型 [${selectedModel.value.name}] 进行预测...`);

  // 1. 按需连接WebSocket
  //    `connect` 方法应该被设计为如果已连接则直接返回成功
  await websocketService.connect();
  console.log('WebSocket 已连接，准备发起任务请求...');

  // 2. 调用API，获取任务ID
  const response = await selectedModel.value.apiFn(selectedFile.value, selectedModel.value.model);
  const taskId = response.taskId;
  console.log(`任务已受理，任务ID: ${taskId}`);

  // 3. 订阅该任务的结果主题
  const destination = `/topic/ai-results/${taskId}`;
  console.log(`正在订阅结果主题: ${destination}`);

  websocketService.subscribe(destination, (message) => {
    // 解析收到的消息体
    const result = JSON.parse(message.body);

    if (result.status === 'success') {
      console.log(`收到任务 ${result.taskId} 的成功结果。`);
      // 将Base64数据转换为可显示的URL
      resultImageUrl.value = `data:${result.imageFormat};base64,${result.result}`;
      isLoading.value = false; // 收到结果后才真正结束加载
    } else {
      console.error(`收到任务 ${result.taskId} 的错误: ${result.error}`);
      errorMsg.value = `任务处理失败: ${result.error}`;
      isLoading.value = false; // 收到错误也结束加载
    }
    // 成功或失败后，可以考虑取消订阅以释放资源
    websocketService.unsubscribe(destination);
  });

  // 注意：对于WebSocket流程，isLoading会在收到消息时才变为false
  // 这里不设置 isLoading.value = false;
};

// --- 4. 生命周期钩子 ---

// 组件卸载时，确保断开WebSocket连接
onUnmounted(() => {
  websocketService.disconnect();
});
</script>

<template>
  <div class="predictor-container">
    <h1>模型测试</h1>
    <p>请选择一个模型，并上传一张图片，模型将会返回标注后的结果。</p>

    <div class="controls">
      <div class="control-item">
        <label for="model-select">选择模型：</label>
        <select id="model-select" v-model="selectedModelId">
          <option v-for="model in models" :key="model.id" :value="model.id">
            {{ model.name }}
          </option>
        </select>
      </div>

      <div class="control-item">
        <input type="file" @change="handleFileUpload" accept="image/*" />
      </div>

      <button @click="submitImage" :disabled="isLoading || !selectedFile">
        {{ isLoading ? '正在分析...' : '开始预测' }}
      </button>
    </div>

    <div v-if="isLoading" class="status-info loading">正在处理图片，请稍候...</div>
    <div v-if="errorMsg" class="status-info error">{{ errorMsg }}</div>

    <div class="image-display" v-if="originalImageUrl">
      <div class="image-card">
        <h2>原始图片</h2>
        <img :src="originalImageUrl" alt="Original Upload" />
      </div>

      <div class="image-card" v-if="resultImageUrl">
        <h2>检测结果</h2>
        <img :src="resultImageUrl" alt="Detection Result" />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 样式基本保持不变，为新元素稍作调整 */
.predictor-container {
  max-width: 1200px; margin: 2rem auto; padding: 2rem;
  font-family: sans-serif; text-align: center; background-color: #f9f9f9;
  border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
h1 { color: #2c3e50; }
.controls {
  margin-bottom: 2rem; display: flex; justify-content: center;
  align-items: center; gap: 1.5rem; flex-wrap: wrap;
}
.control-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
label {
  font-weight: bold;
  color: #333;
}
select, input[type="file"] {
  border: 1px solid #ccc; padding: 0.5rem; border-radius: 4px;
  font-size: 1rem;
}
button {
  padding: 0.75rem 1.5rem; font-size: 1rem; color: white;
  background-color: #42b983; border: none; border-radius: 4px;
  cursor: pointer; transition: background-color 0.3s;
}
button:hover { background-color: #36a374; }
button:disabled { background-color: #ccc; cursor: not-allowed; }
.status-info { margin: 1rem 0; padding: 1rem; border-radius: 4px; }
.loading { color: #31708f; background-color: #d9edf7; }
.error { color: #a94442; background-color: #f2dede; }
.image-display {
  display: flex; justify-content: space-around; gap: 2rem;
  margin-top: 2rem; flex-wrap: wrap;
}
.image-card {
  flex-basis: 45%; min-width: 300px; border: 1px solid #ddd;
  padding: 1rem; border-radius: 8px; background-color: white;
}
.image-card h2 { margin-top: 0; color: #333; }
.image-card img { max-width: 100%; height: auto; border-radius: 4px; }
</style>