"""Implementation of attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.utils import to_categorical
import time
import utils
import os
from scipy import misc
from scipy import ndimage
import PIL
import io

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

'''
tf.flags.DEFINE_string('model_name', 'vgg_16', 'The Model used to generate adv.')

tf.flags.DEFINE_string('layer_name','vgg_16/conv3/conv3_3/Relu','The layer to be attacked.')

tf.flags.DEFINE_string('output_dir', './adv/FIAvgg_16_subfitting/', 'Output directory with images.')

tf.flags.DEFINE_string('attack_method', 'FIA', 'The name of attack method.')

"""parameter for DIM"""
#ince
#tf.flags.DEFINE_integer('image_size', 299, 'size of each input images.')
#vgg,resnet
tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')
'''

tf.flags.DEFINE_string('model_name', 'inception_v3', 'The Model used to generate adv.')

tf.flags.DEFINE_string('layer_name','InceptionV3/InceptionV3/Mixed_5b/concat','The layer to be attacked.')

tf.flags.DEFINE_string('output_dir', './adv/FIAince_v3_delta_diversity30_1000/', 'Output directory with images.')

tf.flags.DEFINE_string('attack_method', 'FIA', 'The name of attack method.')

"""parameter for DIM"""
#ince
tf.flags.DEFINE_integer('image_size', 299, 'size of each input images.')
#vgg,resnet
#tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')

tf.flags.DEFINE_integer('image_resize', 250, 'size of each diverse images.')

tf.flags.DEFINE_float('prob', 0.7, 'Probability of using diverse inputs.')

tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')

""""""

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')

tf.flags.DEFINE_integer('batch_size', 20, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')

"""parameter for TIM"""
tf.flags.DEFINE_integer('Tkern_size', 15, 'Kernel size of TIM.')

"""parameter for PIM"""
tf.flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')

tf.flags.DEFINE_float('gamma', 0.5, 'The gamma parameter.')

tf.flags.DEFINE_integer('Pkern_size', 3, 'Kernel size of PIM.')

"""parameter for FIA"""
tf.flags.DEFINE_float('ens', 30.0, 'Number of random mask input.')

tf.flags.DEFINE_float('probb', 0.9, 'keep probability = 1 - drop probability.')

tf.flags.DEFINE_integer('diversity',30,'diversity')

tf.flags.DEFINE_float('percent', 10, '')

FLAGS = tf.flags.FLAGS
# os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID
#tf.device('/gpu.0')


"""obtain the feature map of the target layer"""
def get_opt_layers(layer_name):
    opt_operations = []
    #shape=[FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3]
    operations = tf.get_default_graph().get_operations()
    #print(operations)
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape

"""the loss function for DFA"""
def get_dfa_loss(opt_operations,weights):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]
        loss += tf.reduce_sum(adv_tensor*weights,axis=(1,2,3))/tf.cast(tf.size(layer[0]),tf.float32)
        #loss -= tf.reduce_sum(tf.square(adv_tensor - ori_tensor),axis=(1,2,3))/tf.cast(tf.size(layer[0]),tf.float32)
        #bbb = tf.size(layer)
        #loss += tf.reduce_sum(adv_tensor*weights) / tf.cast(tf.size(layer), tf.float32)
        #loss += tf.reduce_sum((weights*tf.abs(adv_tensor-ori_tensor))) / tf.cast(tf.size(layer), tf.float32)
    loss = loss / len(opt_operations)
    return loss

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square)
    return nor_grad

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, kern_size):
    x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1, 1, 1, 1], padding='VALID')
    return x

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel

def input_diversity(input_tensor):
    """Input diversity: https://arxiv.org/abs/1803.06978"""
    rnd = tf.random_uniform((), FLAGS.image_size, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret=tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_size, FLAGS.image_size],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

def get_perturbations(diversity,std):
    ortho_with = None
    for i in range(diversity):
        vectors,ortho_with = _get_perturbation(std,ortho_with)
        for j in range(FLAGS.batch_size):
            ortho_with[j] = np.concatenate((ortho_with[j],np.expand_dims(vectors[j],axis=0)),axis=0)
    return ortho_with

def _get_perturbation(std = 0.01,ortho_with = None):
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    if ortho_with is None:
        ortho_with = {i:np.random.normal(0,std,size = batch_shape) for i in range(FLAGS.batch_size)}
    r = _function_generation(std)
    vectors = [_gram_schmidt(r[i],ortho_with[i]) for i in range(FLAGS.batch_size)]
    vectors = np.concatenate([np.expand_dims(v,axis=0) for v in vectors],axis=0)
    norms = np.linalg.norm(vectors.reshape(FLAGS.batch_size,-1),axis=1)
    vectors /= utils.atleast_kdim(norms,len(vectors.shape))
    return vectors,ortho_with

def _function_generation(std = 0.01):
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    return np.random.normal(0,std,size=batch_shape)

def _gram_schmidt(v,ortho_with):
    v_repeated = np.concatenate([np.expand_dims(v,axis=0)] * len(ortho_with),axis=0)
    #print("ortho_with.shape\n",ortho_with.shape)
    #print("v_repeated.shape\n",v_repeated.shape)
    gs_coeff = np.sum((ortho_with * v_repeated).reshape(len(ortho_with),-1),axis=1)
    proj = utils.atleast_kdim(gs_coeff,len(ortho_with.shape)) * ortho_with
    v = v - np.sum(proj,axis=0)
    return v

def modify(weight_np):
    weight_new = np.copy(weight_np.transpose(0, 3, 1, 2))
    shape = weight_new.shape
    batch = shape[0]
    channels = shape[1]
    for i in range(batch):
        for j in range(channels):
            newWeight = weight_new[i][j] #weight*high[][]
            oneTrans = np.abs(newWeight.flatten()) #取完绝对值后的一维数组
            num = np.percentile(oneTrans, FLAGS.percent) 
            weight = newWeight.shape[0]
            high = newWeight.shape[1]
            for k in range(weight):
                for l in range(high):
                    if (newWeight[k][l] > 0 and newWeight[k][l] < num):
                        newWeight[k][l] = 0
                    if (newWeight[k][l] < 0 and -newWeight[k][l] < num):
                        newWeight[k][l] = 0
    weight_new = weight_new.transpose(0,2,3,1)
    return weight_new

P_kern, kern_size = project_kern(FLAGS.Pkern_size)
T_kern = gkern(FLAGS.Tkern_size)

# def softmax(x):
#     x -= np.max(x, axis = 1, keepdims = True)
#     x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
#     return x

def main(_):

    if FLAGS.model_name in ['vgg_16','vgg_19', 'resnet_v1_50','resnet_v1_152']:
        eps = FLAGS.max_epsilon
        alpha = FLAGS.alpha
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0

    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name=FLAGS.layer_name

    with tf.Graph().as_default():
        # Prepare graph
        perturbation = tf.placeholder(tf.float32,shape=batch_shape)
        ori_input  = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input = ori_input + perturbation
        #1001
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size*2,num_classes])
        
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        amplification_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)

        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x=tf.concat([ori_input,adv_input],axis=0)

        # whether using DIM or not
        if 'DI' in FLAGS.attack_method:
            logits, end_points = network_fn(input_diversity(x))
        else:
            logits, end_points = network_fn(x)

        #targeted layer
        opt_operations,shape = get_opt_layers(layer_name)
        weights_ph = tf.placeholder(dtype=tf.float32, shape=shape)

        # select the loss function
        if 'DFA' in FLAGS.attack_method:
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = get_dfa_loss(opt_operations,weights_ph)
        else:
            #problity=tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(logits, axis=1)
            one_hot = tf.one_hot(pred, num_classes)
            entropy_loss = tf.losses.softmax_cross_entropy(one_hot[:FLAGS.batch_size], logits[FLAGS.batch_size:])
            loss = entropy_loss

        gradient=tf.gradients(loss,perturbation)[0]
        
        noise = gradient
        amplification_update = amplification_ph

        # whether using TIM or not
        if 'TI' in FLAGS.attack_method:
            noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')

        # the default optimization process with momentum
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        noise = momentum * accumulated_grad_ph + noise
        #g(t+1)

        # whether using PIM or not
        if 'PI' in FLAGS.attack_method:
            # amplification factor
            alpha_beta = alpha * FLAGS.amplification_factor
            gamma = FLAGS.gamma * alpha_beta

            # Project cut noise
            amplification_update += alpha_beta * tf.sign(noise)
            cut_noise = tf.clip_by_value(abs(amplification_update) - eps, 0.0, 10000.0) * tf.sign(amplification_update)
            projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))

            # Occasionally, when the adversarial examples are crafted for an ensemble of networks with residual block by combined methods,
            # you may neet to comment the following line to get better result.
            amplification_update += projection

            perturbation_update = perturbation + alpha_beta * tf.sign(noise) + projection
        else:
            perturbation_update = perturbation + alpha * tf.sign(noise)
        #x_adv_t+1
        #adv_input_update_norm = tf.sqrt(tf.reduce_sum(tf.square(perturbation_update),axis=(1,2,3)))
        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,checkpoint_path)
            #批量处理所有图片
            count=0
            #perturbation_array = [np.random.normal(0,0.01,size=batch_shape)]
            perturbation_dictionary = get_perturbations(FLAGS.diversity,std = 0.01)
            if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                perturbation_dictionary = get_perturbations(FLAGS.diversity,std = 1.2)
            for images,names,labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size,FLAGS.batch_size):
                count+=FLAGS.batch_size
                if count%100==0:
                    print("Generating:",count)

                images_tmp=image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels=labels-1

                # obtain true label
                labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
                #labels = sess.run(one_hot, feed_dict={ori_input: images_tmp, adv_input: images_tmp})

                for j in range(FLAGS.diversity):
                    grad_np=np.zeros(shape=batch_shape)
                    amplification_np=np.zeros(shape=batch_shape)
                    weight_np = np.zeros(shape=shape)
                    
                    perturbation_input = np.zeros(shape=batch_shape)
                    for item in perturbation_dictionary:
                        perturbation_input[item] = perturbation_dictionary[item][j]
                    #攻击迭代
                    print(j,end=' ')
                    for i in range(num_iter):
                        # calculate the weights(feature importance) for FIA
                        if i==0 and 'DFA' in FLAGS.attack_method:

                            # only use original image to obtain weights
                            if FLAGS.ens == 0:
                                images_tmp2 = image_preprocessing_fn(np.copy(images))
                                w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,label_ph: labels})
                                weight_np =w[:FLAGS.batch_size]

                            # use ensemble masked image to obtain weights
                            for l in range(int(FLAGS.ens)):
                                # generate the random mask
                                mask = np.random.binomial(1, FLAGS.probb, size=(batch_shape[0],batch_shape[1],batch_shape[2],batch_shape[3]))
                                images_tmp2 = images * mask
                                images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))
                                w, feature = sess.run([weights_tensor, opt_operations[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np = weight_np + w[:FLAGS.batch_size]

                            # normalize the weights
                            weight_np = -normalize(weight_np, 2)
                            weight_np = modify(weight_np)
                        #print("*********************",i)
                        # optimization
                        perturbation_input, grad_np, amplification_np,loss_output=sess.run([perturbation_update, noise, amplification_update,loss],
                                              feed_dict={ori_input:images_tmp,perturbation:perturbation_input,weights_ph:weight_np,
                                                         label_ph:labels,accumulated_grad_ph:grad_np,amplification_ph:amplification_np})
                        '''
                        perturbation_input, grad_np, amplification_np,adv_input_update_norm_output,loss_output=sess.run([perturbation_update, noise, amplification_update,adv_input_update_norm,loss],
                                              feed_dict={ori_input:images_tmp,perturbation:perturbation_input,weights_ph:weight_np,
                                                         label_ph:labels,accumulated_grad_ph:grad_np,amplification_ph:amplification_np})
                        '''
                        perturbation_input = np.clip(images_tmp + perturbation_input, images_tmp - eps, images_tmp + eps) - images_tmp
                        #print("adv_input_update_norm_output",adv_input_update_norm_output)
                        #print("FIA loss",loss_output)
                        #print("tf.size(layer)",bbb_output)  #56*56*256 保证数字不太大，容易进行优化
                    images_adv = np.clip(images_tmp + perturbation_input, images_tmp - eps, images_tmp + eps)
                    images_adv = inv_image_preprocessing_fn(images_adv)
                    names_diversity = [name[:-4] + '_' + str(j) + name[-4:] for name in names]
                    utils.save_image(images_adv, names_diversity, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()