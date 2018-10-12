import os
from rbm import RBM
from dbn import DBN

import tensorflow as tf
import input_data
from utilsnn import save_image, min_max_scale, show_image
import matplotlib.pyplot as plt
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
flags.DEFINE_integer('epochs', 50, 'The number of training epochs')
flags.DEFINE_integer('batchsize', 30, 'The batch size')
flags.DEFINE_integer('class_nb', 10, 'Number of classes used in the last DBN layer')
flags.DEFINE_string('out_dir', './out/', 'Directory for storing log')

# ensure output dir exists
if not os.path.isdir(FLAGS.out_dir):
  os.mkdir(FLAGS.out_dir)

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
tr0X, tr0Y, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
tr1X, teX = min_max_scale(tr0X, teX)
# Training and CV sets
trX = tr1X[:50000]
trY = tr0Y[:50000]
cvX = tr1X[50000:]
cvY = tr0Y[50000:]

# RBMs
rbmobject1 = RBM(FLAGS.out_dir, 784, 900, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.3, tf.nn.sigmoid)
rbmobject2 = RBM(FLAGS.out_dir, 900, 500, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.3, tf.nn.sigmoid)

rbmobject1.restore_weights('./out/rbmw1.chp')
rbmobject2.restore_weights('./out/rbmw2.chp')

# DBN
y_colNb=1
dbn = DBN(FLAGS.out_dir, [rbmobject1, rbmobject2], 784, y_colNb,FLAGS.class_nb, 
        FLAGS.batchsize, FLAGS.epochs, 0.3)

#dbn.restore_trained_net('./out/dbn.chp')
#i=8
#dbn.predict(teX[i], teY[i])
#exit()

iterations = len(trX) / FLAGS.batchsize
## Train DBN
print 'Training DBN: ', FLAGS.epochs, ' epochs of ', iterations, ' iterations'
dbn.train(trX, trY, cvX, cvY, teX, teY)
dbn.save_trained_net('./out/dbn.chp')

