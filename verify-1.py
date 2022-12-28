import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import numpy as np
import argparse
import utils
import csv
import shutil
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_names=['inception_v3','inception_v4','inception_resnet_v2','resnet_v1_50','resnet_v1_152',
             'vgg_16','vgg_19','adv_inception_v3','ens3_adv_inception_v3',
             'ens4_adv_inception_v3','adv_inception_resnet_v2','ens_adv_inception_resnet_v2']

# model_names=['vgg_16','resnet_v2_50','resnet_v2_152','vgg_19','adv_inception_v3','adv_inception_resnet_v2',
#             'ens3_adv_inception_v3','ens4_adv_inception_v3','ens_adv_inception_resnet_v2']

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

num = 1000

def verify(model_name,ori_image_path,adv_image_path):
    #os.remove(os.path.join(adv_image_path,'.ipynb_checkpoints'))
    if ".ipynb_checkpoints" in os.listdir(adv_image_path):
        shutil.rmtree(os.path.join(adv_image_path,'.ipynb_checkpoints'))
    checkpoint_path=utils.checkpoint_paths[model_name]
    
    if model_name=='adv_inception_v3' or model_name=='ens3_adv_inception_v3' or model_name=='ens4_adv_inception_v3':
        model_name='inception_v3'
    elif model_name=='adv_inception_resnet_v2' or model_name=='ens_adv_inception_resnet_v2':
        model_name='inception_resnet_v2'

    num_classes=1000+utils.offset[model_name]

    network_fn = utils.nets_factory.get_network_fn(
        model_name,
        num_classes=(num_classes),
        is_training=False)

    image_preprocessing_fn = utils.normalization_fn_map[model_name]
    image_size = utils.image_size[model_name]

    #batch_size = 200
    batch_size=20

    tf.disable_eager_execution()
    image_ph=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_size,image_size,3])

    logits, _ = network_fn(image_ph)
    predictions = tf.argmax(logits, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_path)

        ori_pre=[] # prediction for original images
        adv_pre=[] # prediction label for adversarial images
        ground_truth=[] # grund truth for original images
#         score = np.zeros([5000])
        
        for images,names,labels in utils.load_image(ori_image_path, image_size, batch_size):
            images=image_preprocessing_fn(images)
            pres=sess.run(predictions,feed_dict={image_ph:images})
            ground_truth.extend(labels)
            ori_pre.extend(pres)
#         i = 0
#         j = 20
        for images,names,labels in utils.load_image_diversity(adv_image_path, image_size, batch_size):
#             print(names)
            images=image_preprocessing_fn(images)
            pres=sess.run(predictions,feed_dict={image_ph:images})
            adv_pre.extend(pres)
#             score[i] = len(np.unique(pres))
#             print(j, '-----', score[i])
#             i = i+1
#             j += 20
    tf.reset_default_graph()

    ori_pre=np.array(ori_pre)
    adv_pre=np.array(adv_pre)
    ground_truth=np.array(ground_truth)

    if num_classes==1000:
        ground_truth=ground_truth-1

    return ori_pre,adv_pre,ground_truth

def compute_norm(input_dir,input_adv_dir,batch_size = 20,image_size = 224,diversity=30):

    channels = 3
    batch_shape = [batch_size,image_size,image_size,channels]
    images_all = np.zeros(shape=[num,image_size,image_size,channels])
    norm = 0
    b_i = 0
    for images,labels,filenames in utils.load_image(input_dir,image_size,batch_size):
        bb_i = b_i + batch_size
        images_all[b_i:bb_i] =  images
        b_i = bb_i

    count = 0
    for adv_images,labels,filenames in utils.load_image_diversity(input_adv_dir,image_size,diversity):
        norm += np.mean(np.sum((images_all[count]*diversity-adv_images)**2,axis=(1,2,3))**.5)
        count += 1
        
    norm = np.log10(norm/num)
    #print(norm)
    return norm

def main(ori_path='./dataset/images/',adv_path='./adv/',output_file='./log.csv',diversity=50):

    ori_accuracys=[]
    adv_accuracys=[]
    adv_successrates=[]
    with open(output_file,'a+',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([adv_path])
        writer.writerow(model_names)
        for model_name in model_names:
            print(model_name)
            ori_pre,adv_pre,ground_truth = verify(model_name,ori_path,adv_path)
            #print(ori_pre,adv_pre,ground_truth)
            ori_accuracy = np.sum(ori_pre == ground_truth)/num
            adv_successrate2_array = []
            adv_successrate_array = []
            for i in range(num):
                temp = diversity * [ground_truth[i]] != adv_pre[i*diversity:(i+1)*diversity] 
                #print(np.sum(temp))
                if np.sum(temp) > 0:
                    adv_successrate2_array.append(1)
                else:
                    adv_successrate2_array.append(0)
            adv_successrate2 = np.sum(adv_successrate2_array) / num

            for i in range(num):
                temp = diversity * [ori_pre[i]] != adv_pre[i*diversity:(i+1)*diversity]
#                 print(np.sum(temp))
                if np.sum(temp) > 0:
                    adv_successrate_array.append(1)
                else:
                    adv_successrate_array.append(0)
            adv_successrate = np.sum(adv_successrate_array) / num
            
            print('ori_acc:{:.1%}/adv_suc:{:.1%}/adv_suc2:{:.1%}'.format(ori_accuracy,adv_successrate,adv_successrate2))
            ori_accuracys.append('{:.1%}'.format(ori_accuracy))
            adv_successrates.append('{:.1%}'.format(adv_successrate))
        # print(adv_successrates)
        # writer.writerow(ori_accuracys)
        writer.writerow(adv_successrates)
        writer.writerow([compute_norm(ori_path,adv_path,diversity)])
#         writer.writerow(score)
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ori_path', default='./dataset/images/')
    parser.add_argument('--adv_path',default='./adv/FIA1/')
    parser.add_argument('--output_file', default='./log.csv')
    parser.add_argument('--diversity', default=70)
    args=parser.parse_args()
    main(args.ori_path,args.adv_path,args.output_file,args.diversity)
