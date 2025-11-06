import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from abc import ABC, abstractmethod
import threading  # 导入 threading 模块


class TensorRTBase(ABC):

    def __init__(self, model_path):
        print(f"Loading TensorRT model from: {model_path}")
        # --- 优化点 1: 增加线程锁 ---
        self.lock = threading.Lock()

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        if self.engine is None or self.context is None:
            raise RuntimeError("TensorRT engine或context创建失败")

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [name for name in self.tensor_names if
                            self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self.output_names = [name for name in self.tensor_names if
                             self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

        # --- 优化点 2: 在初始化时创建持久化的 stream ---
        self.stream = cuda.Stream()

        # --- 优化点 3: 使用字典管理缓冲区，更具通用性 ---
        self.host_buffers = {}
        self.device_buffers = {}
        self._alloc_and_bind_buffers()

    @staticmethod
    def _np_dtype_from_trt(dt):
        return {
            trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8, trt.DataType.INT32: np.int32,
            trt.DataType.BOOL: np.bool_,
        }[dt]

    def _alloc_and_bind_buffers(self):
        """为所有 I/O 张量分配锁页内存和设备内存"""
        for name in self.tensor_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._np_dtype_from_trt(self.engine.get_tensor_dtype(name))

            # --- 优化点 4: 使用锁页内存(Pinned Memory)以实现高效异步拷贝 ---
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.host_buffers[name] = host_mem
            self.device_buffers[name] = device_mem
            self.context.set_tensor_address(name, int(device_mem))

    def _maybe_realloc(self, inputs: dict):
        """检查输入尺寸是否变化，如果变化则加锁并重新分配缓冲区"""
        # 以第一个输入为基准检查形状
        first_input_name = self.input_names[0]
        if inputs[first_input_name].shape != self.host_buffers[first_input_name].shape:
            # --- 优化点 1: 在修改共享资源前加锁 ---
            with self.lock:
                # 再次检查，防止在等待锁的过程中其他线程已经分配完毕 (Double-checked locking)
                if inputs[first_input_name].shape != self.host_buffers[first_input_name].shape:
                    print(f"输入形状变化，重新分配缓冲区...")
                    self.context.set_input_shape(first_input_name, inputs[first_input_name].shape)
                    self._alloc_and_bind_buffers()

    @abstractmethod
    def _preprocess(self, input_data, context: dict) -> dict:
        """预处理，必须返回一个 {input_name: tensor} 的字典"""
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, outputs: dict, context: dict) -> any:
        """后处理，接收一个 {output_name: tensor} 的字典"""
        raise NotImplementedError

    def predict(self, data):
        context = {}
        # --- 优化点 5: 接口更加通用和清晰 ---
        input_tensors = self._preprocess(data, context)

        self._maybe_realloc(input_tensors)

        # --- 优化点 6: 完整的异步推理流程 ---
        # 1. 异步将所有输入从 Host 拷贝到 Device
        for name, tensor in input_tensors.items():
            # numpy数组可能不是连续的，拷贝到锁页内存时保证连续性
            np.copyto(self.host_buffers[name], tensor)
            cuda.memcpy_htod_async(self.device_buffers[name], self.host_buffers[name], self.stream)

        # 2. 异步执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 3. 异步将所有输出从 Device 拷贝回 Host
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name], self.stream)

        # 4. 同步CUDA流，等待所有异步操作完成
        self.stream.synchronize()

        # 5. 从host_buffers整理输出字典
        outputs = {name: self.host_buffers[name].copy() for name in self.output_names}

        # 6. 后处理
        return self._postprocess(outputs, context)