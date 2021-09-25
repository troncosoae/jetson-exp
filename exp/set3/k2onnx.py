# import keras2onnx
import tensorflow as tf
import tf2onnx.convert

import onnx


if __name__ == "__main__":
    model = tf.keras.models.load_model(
        '../set2/tmp_saved_models/version1')

    model.summary()

    # onnx_model = keras2onnx.convert_keras(model, model.name)

    # BATCH_SIZE = 64
    # inputs = onnx_model.graph.input
    # for input in inputs:
    #     dim1 = input.type.tensor_type.shape.dim[0]
    #     dim1.dim_value = BATCH_SIZE

    onnx_model, _ = tf2onnx.convert.from_keras(model)

    onnx_model_name = 'onnx_models/version1_onnx.onnx'
    onnx.save(onnx_model, onnx_model_name)
