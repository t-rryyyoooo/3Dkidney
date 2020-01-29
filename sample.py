import tensorflow as tf 
# グラフを作成します。
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# log_device_placement を True にしてセッションを作成します。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# OP を実行します。
print(sess.run(c))