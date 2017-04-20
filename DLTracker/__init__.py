import numpy as np
import tensorflow as tf
import cv2

import time
import sys

class DLTracker :
    fromfile = None
    disp_console = True
    weights_file = None
    alpha = 0.1
    threshold = 0.12
    iou_threshold = 0.5
    num_class = 4
    num_box = 2
    grid_size = 7
    classes =  ["ok", "paper", "rock", "scissor"]
    classID = [3, 1, 0, 2]

    w_img = 640
    h_img = 480

    def __init__(self, weights = None):
        self.weights_file = weights
        self.build_networks()
                
    def build_networks(self):
        if self.disp_console : print( "Building YOLO_tiny graph...")
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
        output_size = 7*7*len(self.classes) + 7*7*2 + 7*7*4*2
        self.fc_19 = self.fc_layer(19,self.do_18,output_size,flat=False,linear=True)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,self.weights_file)
        if self.disp_console : print( "Loading complete!" + '\n')

    def conv_layer(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')
        if self.disp_console : print( '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels)))
        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

    def pooling_layer(self,idx,inputs,size,stride):
        if self.disp_console : print( '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))
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
        if self.disp_console : print( '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	)
        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)
        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')
    
    def dropout_layer(self, idx, inputs):
        if self.disp_console : print( '    Layer  %d : Type = Dropout ' % (idx))
        self.keep_prob = tf.placeholder(tf.float32)
        return tf.nn.dropout(inputs, self.keep_prob)

    def detect_from_cvmat(self,img):
        s = time.time()
        self.h_img,self.w_img,_ = img.shape
        #Resize image and turn into RGB numpy array
        img_resized_np = np.asarray(cv2.cvtColor(cv2.resize(img, (448, 448)),cv2.COLOR_BGR2RGB))
        inputs = np.zeros((1,448,448,3),dtype='float32')
        inputs[0] = (img_resized_np/255.0)*2.0-1.0
        in_dict = {self.x: inputs, self.keep_prob: 1.0}
        net_output = self.sess.run(self.fc_19,feed_dict=in_dict)
        self.res = self.interpret_output(net_output[0])
        self.image = self.show_results(img,self.res)
        strtime = str(time.time()-s)
        #if self.disp_console : print( 'Elapsed time : ' + strtime + ' secs' + '\n')
        if len(self.res) == 0:
            self.result = None
            self.y = 0
        else:
            self.result = self.classID[self.classes.index(self.res[0][0])]
            self.y = self.res[0][2]
            
        return self.result

    def interpret_output(self,output):
        num_classes = len(self.classes)
        probs = np.zeros((7,7,2,num_classes))
        class_probs = np.reshape(output[0:7*7*num_classes],(7,7,num_classes))
        scales = np.reshape(output[7*7*num_classes:7*7*num_classes + 7*7*2],(7,7,2))
        boxes = np.reshape(output[7*7*num_classes + 7*7*2:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
        
        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(num_classes):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
                    probs_filtered[j] = 0.0
                    
        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    def show_results(self,img,results):
        img_cp = img.copy()
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            if self.disp_console : print( '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5]))
            #Add rectangels to image
            cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
            cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            
        return img_cp

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    
if __name__ == "__main__":
    import numpy as np
    import cv2
    
    tracker = DLTracker(weights = '../weights/YOLO_tiny.ckpt')
    
    cap = cv2.VideoCapture(0)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        #Detect objects
        tracker.detect_from_cvmat(frame)
    
        # Display the resulting frame
        cv2.imshow('frame',tracker.image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()