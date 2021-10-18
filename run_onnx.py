import onnx
import onnx_tensorrt.backend as backend
import numpy as np


if __name__ == "__main__":

    PATH = 'saved_nets/saved_onnx/onnx_tf_v1.onnx'

    model = onnx.load(PATH)
    engine = backend.prepare(model, device='CUDA:1')
    input_data = np.random.random(size=(50000, 32, 32, 3)).astype(np.float32)
    output_data = engine.run(input_data)[0]
    print(output_data)
    print(output_data.shape)
