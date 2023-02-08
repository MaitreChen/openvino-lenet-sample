import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("mnist.onnx")

dummy_input = np.random.randn(1, 1, 32, 32).astype(np.float32)
outputs = ort_session.run(None, {"input": dummy_input})
print(outputs[0].shape)
