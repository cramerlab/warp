import tensorflow as tf

gpu_opts = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_opts)
config.allow_soft_placement = True
serialized = config.SerializeToString()
print(list(map(hex, serialized)))