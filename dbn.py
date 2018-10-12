import tensorflow as tf
import numpy as np
from util import next_batch, convertDataToOneHot

class DBN(object):
    def __init__(self, log_dir, rbm_layers, n_features, y_colNb, n_classes,
                  batch_size, epochs, learning_rate=1.0, dropout_param=0.95, 
                  transfer_function=tf.nn.sigmoid, regterm=None):
       
       self.sess = tf.Session()
       self.layers = rbm_layers 
       self.batch_size = batch_size
       self.epochs = epochs
       self.transfer_function = transfer_function
       self.dropout = dropout_param
       self.y_colNb = y_colNb
       self.n_classes = n_classes
       
       # placeholders 
       with tf.name_scope('DBN_input'):
        self.input_data = tf.placeholder(
            tf.float32, [None, n_features], name='x')
        self.y_true = tf.placeholder(
            tf.float32, (None, y_colNb,n_classes), name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep-probs')

        next_train = self.input_data
        self.layer_nodes = []
        self.encoding_w_ = []
        self.encoding_b_ = []
        
        # variables
        # RBM layers
        for l,_ in enumerate(self.layers):
            w_name = 'enc-w-{}'.format(l)
            b_name = 'enc-b-{}'.format(l)
            
            self.encoding_w_.append(tf.Variable(rbm_layers[l].getOWeight(),
                                    name=w_name, dtype=tf.float32))
            self.summary_histo_encoding_w_ = tf.summary.histogram(w_name, self.encoding_w_[l])

            self.encoding_b_.append(tf.Variable(rbm_layers[l].getOhiddenBias(),
                                    name=b_name, dtype=tf.float32))
            self.summary_histo_encoding_b_ = tf.summary.histogram(b_name, self.encoding_b_[l])
            
            with tf.name_scope("encode-{}".format(l)):
                activation = self.transfer_function(tf.add(
                    tf.matmul(next_train, self.encoding_w_[l]),
                    self.encoding_b_[l]))
                # the input to the next layer is the output of this layer
                next_train = tf.nn.dropout(activation, self.keep_prob)
                
            self.layer_nodes.append(next_train)
        
        # additional layer for classification 
        with tf.name_scope('FC_layer'):
            self.in_dim = next_train.get_shape()[1].value
            bound_w=4. * np.sqrt(6. / (self.in_dim + y_colNb* n_classes))
            self.softmax_W = tf.Variable(tf.random_uniform((self.in_dim, y_colNb,n_classes),  
                      -bound_w, bound_w), dtype=tf.float32, name='softmax_W')
            self.softmax_b = tf.Variable(tf.constant(0.1, shape=(y_colNb,n_classes)), 
                    dtype=tf.float32, name='softmax_b')
            self.y_hat = tf.add(tf.tensordot(next_train, self.softmax_W, 1), self.softmax_b)
            self.summary_histo_softmax_W = tf.summary.histogram('softmax_W', self.softmax_W)
            self.summary_histo_softmax_b = tf.summary.histogram('softmax_b', self.softmax_b)
       
        self.layer_nodes.append(self.y_hat)
        
        # loss function
        with tf.name_scope('cost'):
            cost = tf.losses.softmax_cross_entropy(self.y_true, self.y_hat)
            if regterm is not None:
                cost += regterm
                self.summary_cost = tf.summary.scalar('softmax_cross_entropy', cost)

        # optimiser
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        with tf.name_scope('train_step'):
            self.train_step = opt.minimize(cost)
        
        # accuracy
        with tf.name_scope('accuracy'):          
            correct_pred = tf.equal(tf.argmax(self.y_hat, 2),
                                     tf.argmax(self.y_true, 2))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            self.summary_acc = tf.summary.scalar('accuracy', self.accuracy)
       
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.train_writer = tf.summary.FileWriter(log_dir + '/DBN/train', self.sess.graph)
                
    def show_tensor(self, msg, t):
        self.sess.run(tf.Print(t,[t], msg))
        
    # prediction for one image x
    def predict(self, x, y_true=None):            
        x = np.reshape(x, [self.input_data.get_shape()[1].value, 1])
 
        if y_true is not None:
           y = np.reshape(y_true, (1,self.y_colNb, self.n_classes))
           p = self.sess.run(self.y_hat , feed_dict={
                        self.input_data: x.T, self.y_true: y,
                        self.keep_prob: 1})
           self.show_tensor('prediction is ', tf.argmax(p, 2))
           self.show_tensor('truth is ', tf.argmax(y, 2))
        else:
            p = self.sess.run(self.y_hat, feed_dict={
                        self.input_data: x.T,
                        self.keep_prob: 1})
            self.show_tensor('prediction is ', tf.argmax(p, 2))                                     
                    
    def train(self, trX, trY,
                     cvX, cvY,
                     teX=None, teY=None):                         
 
        trData = zip(trX, np.reshape(trY, (len(trY), self.y_colNb, self.n_classes)))
       
        for i  in range(self.epochs):
            for batch in next_batch(trData, self.batch_size):
                batchX, batchY = zip(*batch)
                self.sess.run(
                    self.train_step, feed_dict={
                        self.input_data: batchX,
                        self.y_true: batchY,
                        self.keep_prob: self.dropout})

            print 'epoch ', i
            feed = {self.input_data: cvX,
                    self.y_true: np.reshape(cvY, (len(cvY), self.y_colNb, self.n_classes)),
                    self.keep_prob: 1}
            self.log_train(' CV accuracy: ', feed, i)
            
            if teX is not None:
                feed = {self.input_data: teX,
                    self.y_true: np.reshape(teY, (len(teY), self.y_colNb, self.n_classes)),
                    self.keep_prob: 1}
                self.log_train(' Test accuracy: ', feed, i)

    def log_train(self, msg, feed, step):
         s_acc,s_hew, s_heb, s_hsw, s_hsb, acc = self.sess.run([self.summary_acc, 
                        self.summary_histo_encoding_w_, self.summary_histo_encoding_b_,
                        self.summary_histo_softmax_W, self.summary_histo_softmax_b,
                        self.accuracy], feed_dict=feed)
         self.train_writer.add_summary(s_acc, step)       
         self.train_writer.add_summary(s_hew, step)       
         self.train_writer.add_summary(s_heb, step)       
         self.train_writer.add_summary(s_hsw, step)       
         self.train_writer.add_summary(s_hsb, step)       
         print msg, acc

    def save_trained_net(self, path):
        saver = tf.train.Saver(self.encoding_w_ + self.encoding_b_ 
                                + [self.softmax_W, self.softmax_b],
                                max_to_keep=1)
        save_path = saver.save(self.sess, path)

    def restore_trained_net(self, path):
        saver = tf.train.Saver(self.encoding_w_ + self.encoding_b_ 
                                + [self.softmax_W, self.softmax_b])
        saver.restore(self.sess, path)
