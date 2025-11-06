from abc import ABC, abstractmethod
import onnxruntime as ort

class OnnxBase(ABC):
    """
    模型处理器的抽象基类。
    - 构造函数处理通用的模型加载（ONNX）。
    - predict方法定义了统一的“预处理 -> 推理 -> 后处理”流程。
    - 预处理和后处理方法是抽象的，必须由子类实现。
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        # --- 通用加载逻辑 ---
        print(f"Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    @abstractmethod
    def _preprocess(self, data, context)-> dict:
        """对输入图像进行预处理，必须由子类实现。"""
        pass

    @abstractmethod
    def _postprocess(self, model_outputs, context):
        """对模型输出进行后处理，必须由子类实现。"""
        pass

    def predict(self, data):
        context = {}
        """统一的预测流程。"""
        preprocessed_input = self._preprocess(data, context)

        # --- 通用推理逻辑 ---
        if hasattr(self, 'session'):  # ONNX
            outputs = self.session.run(self.output_names, {name: preprocessed_input[name] for name in self.input_names})
        else:
            raise RuntimeError("Model session not initialized.")

        return self._postprocess(outputs, context)