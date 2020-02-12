import sys
import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tensorflow import name_scope
import argparse
import re
from pathlib import Path
import random
import yaml
import time
import csv
import tensorflow.python.keras.backend as K
from Unet import Construct3DUnetModel



args = None

def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a build 3D_U_Net program')
    parser.add_argument("dataFile", help="Input Dataset file(stracture:data_path label_path)")
    parser.add_argument("-o", "--outfile", help="Output model structure file in YAML format (*.yml).")
    parser.add_argument("-t","--testfile", help="Input Dataset file for validation (stracture:data_path label_path)")
    parser.add_argument("-p", "--patchsize", help="Patch size. (ex. 44x44x28)", default="44x44x28")
    parser.add_argument("-c", "--nclasses", help="Number of classes of segmentaiton including background.", default=3, type=int)
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=30, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size*(Warning:memory use a lot)", default=2, type=int)#orginal default:3
    parser.add_argument("-l", "--learningrate", help="Learning rate", default=1e-4, type=float)
    parser.add_argument("--weightfile", help="The filename of the trained weight parameters file for fine tuning or resuming.")
    parser.add_argument("--initialepoch", help="Epoch at which to start training for resuming a previous training", default=0, type=int)
    parser.add_argument("--logdir", help="Log directory", default='log')
    parser.add_argument("--nobn", help="Do not use batch normalization layer", dest="use_bn", action='store_false')
    parser.add_argument("--nodropout", help="Do not use dropout layer", dest="use_dropout", action='store_false')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args


def main(_):
    #Build 3DU-net

    inputShape = getInputShape(args.dataFile)
    nclasses = args.nclasses
    print("Input shape:", inputshape)
    print("Number of classes:", nclasses)

    inputs = tf.keras.layers.Input(shape=inputshape, name="input")
    segmentation = Construct3DUnetModel(inputs, nclasses, not args.nobn, not args.nodropout)

    model = tf.keras.models.Model(inputs, segmentation,name="3DUnet")
    model.summary()

    #Start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        optimizer = tf.keras.optimizers.Adam(lr=args.learningrate)
        model.compile(loss=penalty_categorical2, optimizer=optimizer, metrics=[dice])
    
    if args.outfile is not None:
        with open(args.outfile, 'w') as f:
            yamlobj = yaml.load(model.to_yaml())
            yaml.dump(yamlobj, f)
            
    #get padding size
    ps = np.array(model.output_shape[1:4])[::-1]
    ips = np.array(model.input_shape[1:4])[::-1]
    paddingsize = ((ips - ps) / 2).astype(np.int)

    #A retraining of interruption
    if args.weightfile is None:
        initial_epoch = 0
    else:
        model.load_weights(args.weightfile, by_name=True)
        initial_epoch = args.initialepoch


    if not os.path.exists(args.logdir+'/model'):
        os.makedirs(args.logdir+'/model')
    latestfile = args.logdir + '/latestweights.hdf5'
    bestfile = args.logdir + '/bestweights.hdf5'
    tb_cbk = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
    best_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=bestfile, save_best_only = True)#, save_weights_only = True)
    latest_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=latestfile)#, save_weights_only = True)
    every_cbk = tf.keras.callbacks.ModelCheckpoint(filepath = args.logdir + '/model/model_{epoch:02d}_{val_loss:.2f}.hdf5')
    callbacks = [tb_cbk,best_cbk,latest_cbk,every_cbk]

    #read dataset
    trainingdatalist = ReadSliceDataList(args.datafile)
    train_data = GenerateBatchData(trainingdatalist, paddingsize, batch_size = args.batchsize)
    if args.testfile is not None:
        testdatalist = ReadSliceDataList(args.testfile)
        #testdatalist = random.sample(testdatalist, int(len(testdatalist)*0.3))
        validation_data = GenerateBatchData(testdatalist, paddingsize, batch_size = args.batchsize)
        validation_steps = len(testdatalist) / args.batchsize
    else:
        validation_data = None
        validation_steps = None

    steps_per_epoch = len(trainingdatalist) // args.batchsize
    print ("Number of samples:", len(trainingdatalist))
    print ("Batch size:", args.batchsize)
    print ("Number of Epochs:", args.epochs)
    print ("Learning rate:", args.learningrate)
    print ("Number of Steps/epoch:", steps_per_epoch)

    #with tf.device('/device:GPU:{}'.format(args.gpuid)):
    historys = model.fit_generator(train_data,
            steps_per_epoch = steps_per_epoch,
            epochs = args.epochs,
            callbacks=callbacks,
            validation_data = validation_data,
            validation_steps = validation_steps,
            initial_epoch = initial_epoch )
            
    
    loss = historys.history['dice']
    val_loss = historys.history['val_dice']
    epochs = len(loss)
    
    history_file = open("history.txt","a")

    for x in range(epochs):
        print("{}\t{}".format(loss[x],val_loss[x]),file = history_file)
    print("\n",file=history_file)
        
    history_file.close()

     tf.keras.backend.clear_session()

    print("\ntime:"+str(t2 - t1))

if __name__ == '__main__':
    args = ParseArgs()
    tf.app.run(main=main, argv=[sys.argv[0]])
