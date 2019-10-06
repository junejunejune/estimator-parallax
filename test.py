import tensorflow as tf
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

if tf.test.gpu_device_name():
  print('default gpu device: {}'.format(tf.test.gpu_device_name()))
else:
  print("plesase install")

#print(tf.test.is_gpu_available())
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
