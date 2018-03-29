import tensorflow as tf
import cv2
import numpy as np
import glob
import time
import os

# We processed image size to be 64
IMAGE_SIZE_h = 64
IMAGE_SIZE_w = 64

# Number of channels: 1 because greyscale
NUM_CHANNELS = 1

# Number of digits
num_digits = 3

# Number of output labels
num_labels = 11

_DEBUG_ = True

def save_debug_img(is_debug, img_path, img):
    """

    :param is_debug:
    :param img_path:
    :param img:
    :return:
    """
    if is_debug:
        cv2.imwrite("./tmp/" + img_path, img)

def get_square_image(input_image, dst_size):
    """
    make image to squared, and resize to dst size
    :param input_image:
    :param dst_size:
    :return: resized image
    """
    height, width = input_image.shape
    if(height > width):
        borderWith = (height - width)/2
        input_image = cv2.copyMakeBorder(input_image, 0, 0, borderWith, borderWith, cv2.BORDER_CONSTANT, None, [255,255,255])
    elif(width > height):
       borderWith = (width - height) / 2
       input_image = cv2.copyMakeBorder(input_image, borderWith, borderWith, 0, 0, cv2.BORDER_CONSTANT, None,[255, 255, 255])

    input_image = cv2.resize(input_image, (dst_size, dst_size), None, 0, 0, cv2.INTER_CUBIC)
    return input_image

def merge_boundingbox(box_1, box_2):
    x_1, y_1, w_1, h_1 = box_1
    x_2, y_2, w_2, h_2 = box_2
    x_new = min(x_1, x_2)
    y_new = min(y_1, y_2)
    x_new_1 = max(x_1 + w_1, x_2 + w_2)
    y_new_1 = max(y_1 + h_1, y_2 + h_2)

    return x_new, y_new, x_new_1 - x_new, y_new_1 - y_new

def process_peiyou(img):

    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mean_value = np.mean(img)
    if mean_value > 128:
        img = 255 - img
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    save_debug_img(_DEBUG_, "binary_morpho.png", img)
    img_copy = img.copy()

    save_debug_img(_DEBUG_, "binary.png", img)
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    for cnt in contours:
        (x, y, w, h) = merge_boundingbox((x,y,w,h), cv2.boundingRect(cnt))
    bouding_img = cv2.rectangle(img, (x,y), (x+w,y+h),[255, 255, 255])
    save_debug_img(_DEBUG_, "bouding.png", bouding_img)
    roi = img_copy[y:y+h, x:x+w]
    border = int(float(h)*0.1)
    roi = cv2.copyMakeBorder(roi, border, border, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
    border = int(float(w)*0.1)
    roi = cv2.copyMakeBorder(roi, 0, 0, border, border,cv2.BORDER_CONSTANT, None, [0, 0, 0])
    roi = 255 - roi
    save_debug_img(_DEBUG_, "roi.png", roi)
    roi = cv2.resize(roi, (IMAGE_SIZE_w, IMAGE_SIZE_h), None, 0, 0, cv2.INTER_CUBIC)
    save_debug_img(_DEBUG_, "resized.png", roi)
    return roi

def get_dst_image_1(input_image, dst_size_h, dst_size_w):
    """
    make image to squared, and resize to dst size
    :param input_image:
    :param dst_size:
    :return: resized image
    """
    ratio = float(dst_size_h) / input_image.shape[0]
    resize_h = int(input_image.shape[0] * ratio)
    resize_w = int(input_image.shape[1] * ratio)

    input_image = cv2.resize(input_image, (resize_w, resize_h), None, 0, 0, cv2.INTER_CUBIC)

    height, width = input_image.shape
    borderWith = (dst_size_w - width)/2
    input_image = cv2.copyMakeBorder(input_image, 0, 0, borderWith, borderWith, cv2.BORDER_CONSTANT, None, [255,255,255])

    input_image = cv2.resize(input_image, (dst_size_w, dst_size_h), None, 0, 0, cv2.INTER_CUBIC)
    return input_image

def get_dst_image(input_image, dst_size):
    """
    make image to squared, and resize to dst size
    :param input_image:
    :param dst_size:
    :return: resized image
    """
    origin_size = int(28 * 0.45)
    ratio = float(origin_size) / input_image.shape[0]
    resize_h = int(input_image.shape[0] * ratio)
    resize_w = int(input_image.shape[1] * ratio)
    input_image = cv2.resize(input_image, (resize_h, resize_w), None, 0, 0, cv2.INTER_CUBIC)

    height, width = input_image.shape
    borderWith = (dst_size - width)/2
    input_image = cv2.copyMakeBorder(input_image, 0, 0, borderWith, borderWith, cv2.BORDER_CONSTANT, None, [255,255,255])

    borderWith = (dst_size - height) / 2
    input_image = cv2.copyMakeBorder(input_image, borderWith, borderWith, 0, 0, cv2.BORDER_CONSTANT, None,[255, 255, 255])
    return input_image

def conv_weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def fc_weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))

def conv_layer(input,  # The previous layer.
               n_channels,  # Num. channels in prev. layer.
               f_size,  # Width and height of each filter.
               n_filters,  # Number of filters.
               weight_name,  # Name of variable containing the weights
               pooling=True):  # Use 2x2 max-pooling.

    # Create weights and biases
    weights = conv_weight_variable(weight_name, [f_size, f_size, n_channels, n_filters])
    biases = bias_variable([n_filters])

    layer = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.nn.relu(layer + biases)

    # Use pooling to down-sample the image resolution?
    if pooling:
        layer = tf.nn.max_pool(layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Return the flattened layer and the number of features.
    return layer_flat, num_features

def fc_layer(input,  # The previous layer.
             num_inputs,  # Num. inputs from prev. layer.
             num_outputs,  # Num. outputs.
             weight_name,  # Name of variable containing the weights
             relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = fc_weight_variable(weight_name, shape=[num_inputs, num_outputs])
    biases = bias_variable([num_outputs])

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if relu:
        layer = tf.nn.relu(layer)

    return layer

def softmax_function(input,  # Previous layer
                     num_inputs,  # Number of inputs from previous layer
                     num_outputs,  # Number of outputs
                     weight_name):  # Name of variable containing the weights

    # Create weights and biases
    weights = fc_weight_variable(weight_name, [num_inputs, num_outputs])
    biases = bias_variable([num_outputs])

    # Softmax
    logits = tf.matmul(input, weights) + biases

    return logits, weights

def model(input):

    # Convolutional Layer 1
    filter_size1 = 3  # Convolution filters are 5 x 5 pixels.
    num_filters1 = 16  # There are 16 of these filters.

    # Convolutional Layer 2
    filter_size2 = 3  # Convolution filters are 5 x 5 pixels.
    num_filters2 = 32  # There are 36 of these filters.

    # Convolutional Layer 3
    filter_size3 = 3  # Convolution filters are 5 x 5 pixels.
    num_filters3 = 64  # There are 48 of these filters.

    # Fully-connected layer
    fc_size = 1024  # Number of neurons in fully-connected layer.

    #Convolutional Layer 1
    conv_1, w_c1 = conv_layer(input, NUM_CHANNELS, filter_size1, num_filters1, 'w_c1', True)

    #Convolutional Layer 2
    conv_2, w_c2 = conv_layer(conv_1, num_filters1, filter_size2, num_filters2, 'w_c2', True)

    #Convolutional Layer 3
    conv_3, w_c3 = conv_layer(conv_2, num_filters2, filter_size3, num_filters3, 'w_c3', False)

    #Droput
    #dropout = tf.nn.dropout(conv_3, keep_prob)

    #Flatten Layer
    #The convolutional layers output 4-dim tensors. We now wish to use these as input in a fully-connected network,
    #which requires for the tensors to be reshaped or flattened to 2-dim tensors.
    flatten, num_features = flatten_layer(conv_3)

    #Check that the tensors now have shape (?, 16384) which means there's an arbitrary number of images which have
    #been flattened to vectors of length 16384 each. Note that 16384 = 16 x 16 x 64

    #Fully-Connected Layer 1
    fc_1 = fc_layer(flatten, num_features, fc_size, 'w_fc1', relu=True)
    #Check that the output of the fully-connected layer is a tensor with shape (?, 128) where the ? means there
    #is an arbitrary number of images and fc_size == 128.

    #Predicted Class
    #The fully-connected layer estimates how likely it is that the input image belongs to each of the 10 classes
    #for each of the 5 digits. However, these estimates are a bit rough and difficult to interpret because the
    #numbers may be very small or large, so we want to normalize them so that each element is limited between
    #zero and one. This is calculated using the so-called softmax function and the result is stored in y_pred.

    logits_0, w_s0 = softmax_function(fc_1, fc_size, num_digits + 2, 'w_s0')
    logits_1, w_s1 = softmax_function(fc_1, fc_size, num_labels, 'w_s1')
    logits_2, w_s2 = softmax_function(fc_1, fc_size, num_labels, 'w_s2')
    logits_3, w_s3 = softmax_function(fc_1, fc_size, num_labels, 'w_s3')
    #logits_4, w_s4 = softmax_function(fc_1, fc_size, num_labels, 'w_s4')
    #logits_5, w_s5 = softmax_function(fc_1, fc_size, num_labels, 'w_s5')
    #y_pred = [logits_1, logits_2, logits_3, logits_4, logits_5]
    y_pred = [logits_1, logits_2, logits_3]
    y_pred_length = tf.argmax(logits_0, dimension=1)

    # The class-number is the index of the largest element.
    y_pred_cls = tf.transpose(tf.argmax(y_pred, dimension=2))

    return y_pred_length, y_pred_cls

class MulDigitRecog:
    """

    """
    def __int__(self):
        self.__sess = None
        self.__output = None
        self.__input_data = None

    def init_model(self, ckpt_name):

        tf.reset_default_graph()
        self.__input_data = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE_h, IMAGE_SIZE_w, NUM_CHANNELS), name='input')
        self.__output = model(self.__input_data)

        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        self.__sess = tf.Session(config=sess_config)
        with self.__sess.as_default():
            saver = tf.train.Saver()
            saver.restore(self.__sess, ckpt_name)

    def __preprocess(self, img):

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not (img.shape[0] == IMAGE_SIZE_h and img.shape[1] == IMAGE_SIZE_w):
            img = process_peiyou(img)
        image = np.zeros([IMAGE_SIZE_h, IMAGE_SIZE_w], dtype=np.float32)
        image[:] = img[:]
        image = np.reshape(image, [1, IMAGE_SIZE_h, IMAGE_SIZE_w, NUM_CHANNELS])
        return image

    def recog_from_file(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return self.recog(img)

    def recog(self, img):
        if img is None:
            return ""
        img = self.__preprocess(img)
        predict = self.__sess.run(self.__output, feed_dict={self.__input_data: img})
        return predict

def save_result_img(img_path, recog_res, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (80, 60), None, 0, 0, cv2.INTER_CUBIC)
    name = os.path.basename(img_path)
    length, digits = recog_res

    font = cv2.FONT_HERSHEY_SIMPLEX
    result = "{}-".format(length[0])
    for digit in digits[0]:
        if digit != 10:
           result += "{}".format(digit)

    font_scale = 0.5
    font_color = (0, 0, 255)
    font_thick = 1
    line_type = cv2.LINE_AA

    result_img = np.zeros( [img.shape[0] + 20, img.shape[1], 3], dtype = np.uint8)
    result_img = 255 - result_img
    result_img[0:img.shape[0],:,:] = img

    pos = (result_img.shape[1]/5, result_img.shape[0] - 5)
    cv2.line(result_img, (0, img.shape[0]), (img.shape[1], img.shape[0]), (0, 0, 0), font_thick, line_type)
    cv2.putText(result_img, result, pos, font, font_scale, font_color, font_thick, line_type)
    cv2.imwrite(save_path + name, result_img)


def recog_img_files(instance, img_path, save_path, f, label=None, ext="*.png"):
    img_file_list = glob.glob(img_path + ext)
    if len(img_file_list) < 1:
        print "no valid image in {}".format(img_path)
        return
    start_time = time.time()
    right = 0
    for img_file in img_file_list:
        recog_res = instance.recog_from_file(img_file)
        print img_file + " ", recog_res
        save_result_img(img_file, recog_res, save_path)
        if label is not None:
           right += (recog_res == int(label))
    duration_time = time.time() - start_time
    str = "avg time: {}ms".format(1000 * duration_time/len(img_file_list))
    f.write(str + "\n")
    print str
    if label is not None:
        str = "right rate {0:.3f}%".format(float(100.0 * right) / len(img_file_list))
        f.write(str)
        print str
    return right,len(img_file_list)

if __name__ == '__main__':

    ckpt_path = "./checkpoints/mnist_net2_1_980000"

    instance = MulDigitRecog()
    instance.init_model(ckpt_name=ckpt_path)

    save_path = "/home/gaolining/host_share/digit/result/"
    img_file_path = "/home/gaolining/host_share/digit/samples_m/3-big/"
    #img_file_path = "/home/gaolining/host_share/digit/test_data/3/"
    #img_file_path = "/home/extend/code/models/official/mnist/"

    f = open("log.txt", "w")
    recog_img_files(instance, img_file_path, save_path, f)
    f.close()