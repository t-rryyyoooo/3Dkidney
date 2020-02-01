import tensorflow as tf

def dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice



def penalty_categorical2(y_true,y_pred):
    array_tf = tf.convert_to_tensor(y_true,dtype=tf.float32)
    pred_tf = tf.convert_to_tensor(y_pred,dtype=tf.float32)

    epsilon = K.epsilon()

    result = tf.reduce_sum(array_tf,[0,1,2,3])

    result_pow = tf.pow(result,1.0/3.0)
    weight_y = result_pow / tf.reduce_sum(result_pow)

    return (-1) * tf.reduce_sum( 1 / (weight_y + epsilon) * array_tf * tf.log(pred_tf + epsilon),axis=-1)
 
