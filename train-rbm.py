import os
from rbm import RBM
from au import AutoEncoder
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
flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')
flags.DEFINE_string('out_dir', './out/', 'Directory for storing log')

# ensure output dir exists
if not os.path.isdir(FLAGS.out_dir):
  os.mkdir(FLAGS.out_dir)

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX, teX = min_max_scale(trX, teX)

#save_image("out/teX.jpg", teX, (28, 28),(100,100))
#show_image("out/teX.jpg")
#exit()

sess = tf.Session()

# RBMs
rbmobject1 = RBM(sess, FLAGS.out_dir, 784, 900, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.3, tf.nn.sigmoid)
#, tf.nn.relu)
rbmobject2 = RBM(sess, FLAGS.out_dir, 900, 500, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.3, tf.nn.sigmoid)
#, tf.nn.relu)

init = tf.global_variables_initializer()
sess.run(init)

if FLAGS.restore_rbm:
  rbmobject1.restore_weights('./out/rbmw1.chp')
  rbmobject2.restore_weights('./out/rbmw2.chp')
 

iterations = len(trX) / FLAGS.batchsize

# Train First RBM
print 'first rbm: ', FLAGS.epochs, ' epochs of ', iterations, ' iterations'
for i in range(FLAGS.epochs): 
  print 'epoch=', i
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    _,batch_err=rbmobject1.partial_fit(batch_xs)
rbmobject1.save_weights('./out/rbmw1.chp')
save_image("out/1rbm.jpg", rbmobject1.n_w.T, (28, 28), (100, 100))
show_image("out/1rbm.jpg")
rbmobject1.end_training()
#
# Train Second RBM
rbmobject1.restore_weights('./out/rbmw1.chp')
# improves init
w1=rbmobject1.getOWeight()
rbmobject2.setOWeight(np.transpose(w1))
print 'second rbm: ', FLAGS.epochs, ' epochs of ', iterations, ' iterations'
for i in range(FLAGS.epochs):
  print 'epoch=', i
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    # Transform features with first rbm for second rbm
    batch_xs = rbmobject1.sample_v2h(batch_xs)
    _,batch_err=rbmobject2.partial_fit(batch_xs)
rbmobject2.save_weights('./out/rbmw2.chp')
show_image("out/2rbm.jpg")
rbmobject2.end_training()


