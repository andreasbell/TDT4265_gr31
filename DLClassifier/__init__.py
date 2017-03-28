import tensorflow as tf
from os import listdir
import numpy as np
import random
import cv2

def out(x):
    return {
        'rock': np.array([1, 0, 0, 0], dtype='float32'),
        'paper': np.array([0, 1, 0, 0], dtype='float32'),
        'scissor': np.array([0, 0, 1, 0], dtype='float32'),
        'ok': np.array([0, 0, 0, 1], dtype='float32'),
    }[x]

def generate_batch(size, test = False):
    if test: 
        sample = random.sample(test_data, size)
    else:
        sample = random.sample(training_data, size)
        
    inputs = np.zeros((size,448,448,3),dtype='float32')
    outputs = np.zeros((size,4),dtype='float32')
    for i, elm in enumerate(sample):
        img = cv2.imread(elm[1])
        img_resized_np = np.asarray(cv2.cvtColor(cv2.resize(img, (448, 448)),cv2.COLOR_BGR2RGB))
        inputs[i] = (img_resized_np/255.0)*2.0-1.0
        outputs[i] = out(elm[0])
    return [inputs, outputs]


class DLClassifier:
    # member variabels
    weights_file = None
    alpha = 0.1
    var = []
    
    #Member functions 
    def __init__(self, weights = None):
        self.weights_file = weights
        self.build_network()
        
    def build_network(self):
        tf.reset_default_graph()
        self.x = tf.placeholder('float32',[None,448,448,3])
        self.conv_1 = self.conv_layer(1,self.x,16,3,1)
        self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)
        self.conv_3 = self.conv_layer(3,self.pool_2,32,3,1)
        self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
        self.conv_5 = self.conv_layer(5,self.pool_4,64,3,1)
        self.pool_6 = self.pooling_layer(6,self.conv_5,2,2)
        self.conv_7 = self.conv_layer(7,self.pool_6,128,3,1)
        self.pool_8 = self.pooling_layer(8,self.conv_7,2,2)
        self.conv_9 = self.conv_layer(9,self.pool_8,256,3,1)
        self.pool_10 = self.pooling_layer(10,self.conv_9,2,2)
        self.conv_11 = self.conv_layer(11,self.pool_10,512,3,1)
        self.pool_12 = self.pooling_layer(12,self.conv_11,2,2)
        self.conv_13 = self.conv_layer(13,self.pool_12,1024,3,1)
        self.conv_14 = self.conv_layer(14,self.conv_13,1024,3,1)
        self.conv_15 = self.conv_layer(15,self.conv_14,1024,3,1)
        self.fc_16 = self.fc_layer(16,self.conv_15,256,flat=True,linear=False)
        self.fc_17 = self.fc_layer(17,self.fc_16,4096,flat=False,linear=False)
        self.do_18 = self.dropout_layer(18, self.fc_17)
        self.fc_19 = self.fc_layer(19,self.do_18,4,flat=False,linear=True)
        self.sm_20 = tf.nn.softmax(self.fc_19, name = "Output_prediction")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.weights_file is not None:
            self.saver.restore(self.sess,self.weights_file)
            pass
        print("Graph created")
    
    def conv_layer(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')
        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')
    
    def pooling_layer(self,idx,inputs,size,stride):
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

    def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(inputs,(0,3,1,2))
            inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
            
        weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        
        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)
        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')
    
    def dropout_layer(self, idx, inputs):
        self.keep_prob = tf.placeholder(tf.float32)
        return tf.nn.dropout(inputs, self.keep_prob)
    
    def detect_from_cvmat(self,img):
        #Resize image and turn into RGB numpy array
        img_resized_np = np.asarray(cv2.cvtColor(cv2.resize(img, (448, 448)),cv2.COLOR_BGR2RGB))
        inputs = np.zeros((1,448,448,3),dtype='float32')
        inputs[0] = (img_resized_np/255.0)*2.0-1.0
        net_output = self.sess.run(self.sm_20, feed_dict={self.x: inputs, self.keep_prob: 1.0})
        self.result = net_output[0]
        return self.result
    
    def train_network(self, path):
        
        var = tf.global_variables()
        
        y = tf.placeholder(tf.float32, [None, 4])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.sm_20 ))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list = var[16:]) #var[18:]
        
        correct_prediction = tf.equal(tf.argmax(self.sm_20,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        
        with self.sess.as_default():
            self.saver.restore(self.sess, path)
            print("Model restored.")
            for i in range(0, 1000000):
                batch = generate_batch(25)
                print("Batch: " + str(i))
                if i%20 == 0:
                    save_path = self.saver.save(self.sess, path)
                    test_batch = generate_batch(100, test = True)
                    train_accuracy = accuracy.eval(feed_dict={self.x:test_batch[0], y: test_batch[1], self.keep_prob: 1.0})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={self.x: batch[0], y: batch[1], self.keep_prob: 0.5})
            print("Training finished!")

    

if __name__ == "__main__":
    det = DLClassifier("weights/detector.ckpt")

    #Set up training and test data
    data_path = "D:/data/Images"
    data = []
    for c in listdir(data_path):
        for img in listdir(data_path + '/' + c):
            data.append([c, data_path + '/' + c + '/' + img])
    
    test_data =  random.sample(data, 100)
    training_data =  [d for d in data if not d in test_data]

    #Train network
    det.train_network("weights/detector.ckpt")
