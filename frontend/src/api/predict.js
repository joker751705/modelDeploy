import api from "@/api/index";

export async function predictImageWithWebSocket(imageFile, model){
    const formData = new FormData();
    formData.append('image', imageFile);

    // 假设你的后端通过路径参数区分模型
    const response = await api.post(`/ai/process-image/${model}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
}

export async function predictImageWithHttp(imageFile, model) {
    // 创建一个 FormData 对象，因为我们需要上传文件
    const formData = new FormData();
    formData.append('image', imageFile);

    try {
        const response = await api.post(`/ai/image/${model}`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            responseType: 'blob'
        });

        // response.data 现在是一个 Blob 对象
        return response.data;
    } catch (error) {
        console.error("API request failed:", error);
        // 抛出错误，让组件可以捕获并处理
        throw error;
    }
}