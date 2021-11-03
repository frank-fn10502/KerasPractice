import tensorflow as tf
from tensorflow.keras import layers



# tensorflow.python.framework.ops.EagerTensor' object has no attribute '_keras_history'
#為了解決以上的錯誤(主要是畫圖時會需要用到該屬性)
class StochasticDropout(layers.Layer):
    # from effNetV2 code #survival_prob 預設 0.8 def drop_connect(inputs, is_training = True, survival_prob = 0.8):
    def call(self, inputs):
        """Drop the entire conv with given survival probability."""
        # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
        is_training = True
        survival_prob = 0.8
        if not is_training:
            return inputs

        # Compute tensor.
        batch_size = tf.shape(inputs)[0]
        random_tensor = survival_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        # Unlike conventional way that multiply survival_prob at test time, here we
        # divide survival_prob at training time, such that no addition compute is
        # needed at test time.
        output = inputs / survival_prob * binary_tensor
        return output
