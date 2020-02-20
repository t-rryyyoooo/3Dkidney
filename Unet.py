import tensorflow as tf

def CreateConv3DBlock(x, filters, n = 2, use_bn = True, apply_pooling = True, name = 'convblock'):
    for i in range(n):
        x = tf.compat.v1.keras.layers.Conv3D(filters[i], (3,3,3), padding='same', name=name+'_conv'+str(i+1))(x)
        if use_bn:
            x = tf.compat.v1.keras.layers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
        x = tf.compat.v1.keras.layers.Activation('relu', name=name+'_relu'+str(i+1))(x)
    
    convresult = x

    if apply_pooling:
        x = tf.compat.v1.keras.layers.MaxPool3D(pool_size=(2,2,2), name=name+'_pooling')(x)

    return x, convresult


def CreateUpConv3DBlock(x, contractpart, filters, n = 2, use_bn = True, name = 'upconvblock'):
    # upconv x
    x = tf.compat.v1.keras.layers.Conv3DTranspose((int)(x.shape[-1]), (2,2,2), strides=(2,2,2), padding='same', use_bias = False, name=name+'_upconv')(x)

    x = tf.compat.v1.keras.layers.concatenate([contractpart, x])

    # conv x 2 times
    for i in range(n):
        x = tf.compat.v1.keras.layers.Conv3D(filters[i], (3,3,3), padding='same', name=name+'_conv'+str(i+1))(x)
        if use_bn:
            x = tf.compat.v1.keras.layers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
        x = tf.compat.v1.keras.layers.Activation('relu', name=name+'_relu'+str(i+1))(x)

    return x

def Construct3DUnetModel(input_images, nclasses, use_bn = True, use_dropout = True):
    with tf.name_scope("contract1"):
        x, contract1 = CreateConv3DBlock(input_images, (32, 64), n = 2, use_bn = use_bn, name = 'contract1')

    with tf.name_scope("contract2"):
        x, contract2 = CreateConv3DBlock(x, (64, 128), n = 2, use_bn = use_bn, name = 'contract2')

    with tf.name_scope("contract3"):
        x, contract3 = CreateConv3DBlock(x, (128, 256), n = 2, use_bn = use_bn, name = 'contract3')

    with tf.name_scope("contract4"):
        x, _ = CreateConv3DBlock(x, (256, 512), n = 2, use_bn = use_bn, apply_pooling = False, name = 'contract4')

    with tf.name_scope("dropout"):
        if use_dropout:
            x = tf.compat.v1.keras.layers.Dropout(0.5, name='dropout')(x)

    with tf.name_scope("expand3"):
        x = CreateUpConv3DBlock(x, contract3, (256, 256), n = 2, use_bn = use_bn, name = 'expand3')

    with tf.name_scope("expand2"):
        x = CreateUpConv3DBlock(x, contract2, (128, 128), n = 2, use_bn = use_bn, name = 'expand2')

    with tf.name_scope("expand1"):
        x = CreateUpConv3DBlock(x, contract1, (64, 64), n = 2, use_bn = use_bn, name = 'expand1')

    with tf.name_scope("segmentation"):
        layername = 'segmentation_{}classes'.format(nclasses)
        x = tf.compat.v1.keras.layers.Conv3D(nclasses, (1,1,1), activation='softmax', padding='same', name=layername)(x)

    return x

