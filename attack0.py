"""Implementation of attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import utils
import os
from scipy import misc
from scipy import ndimage
from PIL import Image
import io
import darkon

slim = tf.contrib.slim

# tf.flags.DEFINE_string('attack_method', 'FIA', 'The name of attack method.')

# tf.flags.DEFINE_string('model_name', 'vgg_16', 'The Model used to generate adv.')

# tf.flags.DEFINE_string('layer_name','vgg_16/conv3/conv3_3/Relu','The layer to be attacked.')

# tf.flags.DEFINE_string('output_dir', './adv/FIA/', 'Output directory with images.')

tf.flags.DEFINE_string('attack_method', 'FIA', 'The name of attack method.')

tf.flags.DEFINE_string('model_name', 'inception_v3', 'The Model used to generate adv.')

tf.flags.DEFINE_string('layer_name','InceptionV3/InceptionV3/Mixed_5b/concat','The layer to be attacked.')

tf.flags.DEFINE_string('output_dir', './adv/TFA/', 'Output directory with images.')

tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string('valid_dir', '../PSBA-master/raw_data/imagenet/ILSVRC2012_img_val/', 'Output directory with images.')

tf.flags.DEFINE_string('label_csv', "./dataset/dev_dataset.csv", 'label information with csv file.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')

tf.flags.DEFINE_integer('batch_size', 20, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')

"""parameter for DIM"""
tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')

tf.flags.DEFINE_integer('image_resize', 250, 'size of each diverse images.')

tf.flags.DEFINE_float('prob', 0.7, 'Probability of using diverse inputs.')

"""parameter for TIM"""
tf.flags.DEFINE_integer('Tkern_size', 15, 'Kernel size of TIM.')

"""parameter for PIM"""
tf.flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')

tf.flags.DEFINE_float('gamma', 0.5, 'The gamma parameter.')

tf.flags.DEFINE_integer('Pkern_size', 3, 'Kernel size of PIM.')

"""parameter for FIA"""
tf.flags.DEFINE_float('ens', 0, 'Number of random mask input.')

tf.flags.DEFINE_float('probb', 0.9, 'keep probability = 1 - drop probability.')

"""parameter for PullPush"""
tf.flags.DEFINE_float('interval', 10, 'keep probability = 1 - drop probability.')

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

"""obtain the feature map of the target layer"""
def get_opt_layers(layer_name):
    opt_operations = []
    #shape=[FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3]
    operations = tf.compat.v1.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape

"""the loss function for FDA"""
def get_fda_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        batch_size = FLAGS.batch_size
        tensor = layer[:batch_size]
        mean_tensor = tf.stack([tf.reduce_mean(tensor, -1), ] * tensor.shape[-1], -1)
        wts_good = tensor < mean_tensor
        wts_good = tf.to_float(wts_good)
        wts_bad = tensor >= mean_tensor
        wts_bad = tf.to_float(wts_bad)
        loss += tf.log(tf.nn.l2_loss(wts_good * (layer[batch_size:batch_size*2]) / tf.cast(tf.size(layer),tf.float32)))
        loss -= tf.log(tf.nn.l2_loss(wts_bad * (layer[batch_size:batch_size*2]) / tf.cast(tf.size(layer),tf.float32)))
    loss = loss / len(opt_operations)
    return loss

"""the loss function for NRDM"""
def get_nrdm_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:FLAGS.batch_size*2]
        loss+=tf.norm(ori_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
    loss = loss / len(opt_operations)
    return loss

"""the loss function for FIA"""
def get_fia_loss(opt_operations,weights):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:FLAGS.batch_size*2]
        loss += tf.reduce_sum(adv_tensor*weights) / tf.cast(tf.size(layer), tf.float32)
        #loss += tf.reduce_sum((weights*tf.abs(adv_tensor-ori_tensor))) / tf.cast(tf.size(layer), tf.float32)
    loss = loss / len(opt_operations)
    return loss

def get_tfa_loss(opt_operations,weights_tgt_placeholder,weights_ori_placeholder):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:FLAGS.batch_size*2]
        loss -= tf.reduce_sum(adv_tensor*weights_ori_placeholder)/tf.cast(tf.size(layer),tf.float32)
        loss += tf.reduce_sum(adv_tensor*weights_tgt_placeholder)/tf.cast(tf.size(layer),tf.float32)
    loss = loss / len(opt_operations)
    return loss

def get_pp_loss(opt_operations,interval):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size] #ori
        adv_tensor = layer[FLAGS.batch_size:FLAGS.batch_size*2]
        tgt_tensor = layer[FLAGS.batch_size*2:] #tgt
        loss += tf.norm(tgt_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
        loss -= tf.norm(ori_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
        loss1 = tf.norm(tgt_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
        loss2 = tf.norm(ori_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
    loss = loss / len(opt_operations)
    return loss,loss1,loss2

def get_nrdm_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:FLAGS.batch_size*2]
        loss+=tf.norm(ori_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
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

#实现batch图片的卷积——深度卷积
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

P_kern, kern_size = project_kern(FLAGS.Pkern_size)
T_kern = gkern(FLAGS.Tkern_size)

# def softmax(x):
#     x -= np.max(x, axis = 1, keepdims = True)
#     x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
#     return x

#target_images = get_target_images(target_labels,FLAGS.valid_dir)
def get_target_images(target_labels,valid_dir,batch_shape):
    #在5万张图片中选labels对应的images
    #标签排序后选第一张
    path = r'.\dataset\valid_gt.csv'
    with open(path) as f:
        data = np.loadtxt(f,str,delimiter=',',skiprows=1)

    data = data[np.argsort(np.array(data[:,1],dtype=np.int))]
    if FLAGS.model_name not in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
        target_labels = target_labels - 1
    images = np.zeros(shape=batch_shape)
    for i in range(batch_shape[0]):
        filename = data[target_labels[i]*50+4][0]
        image=Image.open(valid_dir + filename + '.JPEG')
        image=image.resize((batch_shape[1],batch_shape[1]))
        #print(np.array(image).shape)
        #print(filename)
        images[i]=np.array(image)
        
    return images

def main(_):

    if FLAGS.model_name in ['vgg_16','vgg_19', 'resnet_v1_50','resnet_v1_152']:
        eps = FLAGS.max_epsilon
        alpha = FLAGS.alpha
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0
    FLAGS.image_size=utils.image_size[FLAGS.model_name]
    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name=FLAGS.layer_name

    with tf.Graph().as_default():

        # Prepare graph
        ori_input = tf.placeholder(tf.float32, shape=batch_shape)
        tgt_input = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size*3,num_classes])
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        amplification_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)

        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x=tf.concat([ori_input,adv_input,tgt_input],axis=0)

        # single_shape = [1,FLAGS.image_size, FLAGS.image_size, 3]
        # single_input = tf.placeholder(tf.float32,shape=single_shape)
        # W_s_input = tf.placeholder(tf.float32,shape=batch_shape)
        # logits_single, end_points = network_fn(single_input)
        # graph = tf.get_default_graph()
        # insp = darkon.Gradcam(single_input,num_classes,layer_name,graph=graph)
        # prob = insp._prob_ts

        # whether using DIM or not
        if 'DI' in FLAGS.attack_method:
            logits, end_points = network_fn(input_diversity(x))
        else:
            logits, end_points = network_fn(x)

        problity=tf.nn.softmax(logits,axis=1)
        pred = tf.argmax(logits, axis=1)
        one_hot = tf.one_hot(pred, num_classes)

        entropy_loss = tf.compat.v1.losses.softmax_cross_entropy(one_hot[:FLAGS.batch_size], logits[FLAGS.batch_size:FLAGS.batch_size*2])

        opt_operations,shape = get_opt_layers(layer_name)
        weights_tgt_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)
        weights_ori_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)

        # select the loss function
        if 'FDA' in FLAGS.attack_method:
            loss = get_fda_loss(opt_operations)
        elif 'NRDM' in FLAGS.attack_method:
            loss = get_nrdm_loss(opt_operations)
        elif 'FIA' in FLAGS.attack_method:
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = -get_fia_loss(opt_operations,weights_tgt_placeholder)
        elif 'TFA' in FLAGS.attack_method:
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = -get_tfa_loss(opt_operations,weights_tgt_placeholder,weights_ori_placeholder)
        elif 'PP' in FLAGS.attack_method:
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss,loss1,loss2 = get_pp_loss(opt_operations,FLAGS.interval)
            loss = -loss
        else:
            loss = -entropy_loss

        #W_s


        #############
        gradient=tf.gradients(loss,adv_input)[0]

        noise = gradient
        adv_input_update = adv_input
        amplification_update = amplification_ph

        # whether using TIM or not
        if 'TI' in FLAGS.attack_method:
            noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')

        # the default optimization process with momentum
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        noise = momentum * accumulated_grad_ph + noise

        # whether using PIM or not
        if 'PI' in FLAGS.attack_method:
            # amplification factor
            alpha_beta = alpha * FLAGS.amplification_factor
            gamma = FLAGS.gamma * alpha_beta

            # Project cut noise
            amplification_update += alpha_beta * tf.sign(noise)
            cut_noise = tf.clip_by_value(abs(amplification_update) - eps, 0.0, 10000.0) * tf.sign(amplification_update)
            projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))#*W_s_input)

            # Occasionally, when the adversarial examples are crafted for an ensemble of networks with residual block by combined methods,
            # you may neet to comment the following line to get better result.
            amplification_update += projection

            adv_input_update = adv_input_update + alpha_beta * tf.sign(noise) + projection
        else:
            adv_input_update = adv_input_update + alpha * tf.sign(noise)

        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,checkpoint_path)
            true_label_list,target_label_list = utils.label_dict(FLAGS.label_csv)
            count=0
            for images,names,labels,target_labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size,FLAGS.batch_size,true_label_list,target_label_list):
                count+=FLAGS.batch_size
                # if count%100==0:
                #     print("Generating:",count)
                print("Generating:",count)
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels=labels-1
                    target_labels = target_labels - 1

                #get_target_images
                target_images = get_target_images(np.copy(target_labels),FLAGS.valid_dir,batch_shape)

                # obtain true label将整型的类别标签转为onehot编码
                labels= to_categorical(np.concatenate([target_labels,labels,labels],axis=-1),num_classes)
                #labels = sess.run(one_hot, feed_dict={ori_input: images_tmp, adv_input: images_tmp})

                #add some noise to avoid F_{k}(x)-F_{k}(x')=0
                if 'NRDM' in FLAGS.attack_method or 'PP' in FLAGS.attack_method:
                    images_adv=images+np.random.normal(0,0.1,size=np.shape(images))
                else:
                    images_adv=images

                images_adv=image_preprocessing_fn(np.copy(images_adv))
                images_tmp_target=image_preprocessing_fn(np.copy(target_images))
                images_tmp=image_preprocessing_fn(np.copy(images))

                grad_np=np.zeros(shape=batch_shape)
                amplification_np=np.zeros(shape=batch_shape)
                #weight_np = np.zeros(shape=shape)
                weight_tgt = np.zeros(shape=shape)
                weight_ori = np.zeros(shape=shape)

                W_s = np.zeros(shape=batch_shape)

                for i in range(num_iter):
                    # calculate the weights(feature importance) for FIA
                    if i==0 and ('FIA' in FLAGS.attack_method or 'TFA' in FLAGS.attack_method):

                        # only use original image to obtain weights
                        if FLAGS.ens == 0:
                            images_tmp2 = image_preprocessing_fn(np.copy(images))
                            images_tmp2_target = image_preprocessing_fn(np.copy(target_images))
                            w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,tgt_input:images_tmp2_target,label_ph: labels})
                            weight_tgt = w[FLAGS.batch_size*2:]
                            weight_ori = w[FLAGS.batch_size:FLAGS.batch_size*2]

                        # # use ensemble masked image to obtain weights
                        # for l in range(int(FLAGS.ens)):
                        #     # generate the random mask
                        #     mask = np.random.binomial(1, FLAGS.probb, size=(batch_shape[0],batch_shape[1],batch_shape[2],batch_shape[3]))
                        #     images_tmp2 = images * mask
                        #     images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))
                        #     w, feature = sess.run([weights_tensor, opt_operations[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                        #     weight_np = weight_np + w[:FLAGS.batch_size]

                        # normalize the weights
                        weight_tgt = -normalize(weight_tgt, 2)
                        weight_ori = -normalize(weight_ori, 2)
                    # if i==0 and 'PI' in FLAGS.attack_method:

                    #     for i in range(FLAGS.batch_size):
                    #         prob_eval = sess.run(prob,feed_dict={single_input:np.expand_dims(images_tmp[i],axis=0)})
                    #         top0_result = np.argmax(prob_eval,axis=1)
                    #         W_s0 = insp.gradcam(sess,images_tmp[0],top0_result+1-utils.offset[FLAGS.model_name])['heatmap']
                    #         W_s[i] = np.stack([W_s0, W_s0, W_s0]).swapaxes(0,2)


                    # optimization
                    # images_adv, grad_np, amplification_np,loss_output,loss_output1,loss_output2 = sess.run([adv_input_update, noise, amplification_update,loss,loss1,loss2],
                    #                           feed_dict={ori_input:images_tmp,adv_input:images_adv,tgt_input:images_tmp_target,weights_tgt_placeholder:weight_tgt,weights_ori_placeholder:weight_ori,
                    #                                      label_ph:labels,accumulated_grad_ph:grad_np,amplification_ph:amplification_np,W_s_input:W_s})

                    images_adv, grad_np, amplification_np,loss_output= sess.run([adv_input_update, noise, amplification_update,loss],
                                              feed_dict={ori_input:images_tmp,adv_input:images_adv,tgt_input:images_tmp_target,weights_tgt_placeholder:weight_tgt,weights_ori_placeholder:weight_ori,
                                                         label_ph:labels,accumulated_grad_ph:grad_np,amplification_ph:amplification_np})
                    images_adv = np.clip(images_adv, images_tmp - eps, images_tmp + eps)
                    print('loss___',loss_output)
                    # print('loss1___',loss_output1)
                    # print('loss2___',loss_output2)
                images_adv = inv_image_preprocessing_fn(images_adv)
                utils.save_image(images_adv, names, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()
