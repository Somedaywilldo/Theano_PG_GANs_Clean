#coding=utf-8

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import time
import glob
import shutil
import operator
import numpy as np
import scipy.ndimage

import misc
misc.init_output_logging()

if __name__ == "__main__":
    print 'Importing Theano...'

import config
os.environ['THEANO_FLAGS'] = ','.join([key + '=' + value for key, value in config.theano_flags.iteritems()])
sys.setrecursionlimit(10000)
import theano
from theano import tensor as T
import lasagne

import network
import dataset

#----------------------------------------------------------------------------
# Convenience.

def Tsum (*args, **kwargs): return T.sum (*args, dtype=theano.config.floatX, acc_dtype=theano.config.floatX, **kwargs)
def Tmean(*args, **kwargs): return T.mean(*args, dtype=theano.config.floatX, acc_dtype=theano.config.floatX, **kwargs)

def adam(loss, params, **kwargs):
    connected_params = []
    connected_grads = []
    for p in params:
        try:
            g = theano.grad(loss, p)
            connected_params.append(p)
            connected_grads.append(g)
        except theano.gradient.DisconnectedInputError:
            pass
    return lasagne.updates.adam(connected_grads, connected_params, **kwargs)

def random_latents(num_latents, G_input_shape):
    return np.random.randn(num_latents, *G_input_shape[1:]).astype(np.float32)

def random_labels(num_labels, training_set):
    return training_set.labels[np.random.randint(training_set.labels.shape[0], size=num_labels)]

def load_dataset(dataset_spec=None, verbose=False, **spec_overrides):
    if verbose: print 'Loading dataset...'
    if dataset_spec is None: dataset_spec = config.dataset
    dataset_spec = dict(dataset_spec) # take a copy of the dict before modifying it
    dataset_spec.update(spec_overrides)
    dataset_spec['h5_path'] = os.path.join(config.data_dir, dataset_spec['h5_path'])
    if 'label_path' in dataset_spec: dataset_spec['label_path'] = os.path.join(config.data_dir, dataset_spec['label_path'])
    training_set = dataset.Dataset(**dataset_spec)
    if verbose: print 'Dataset shape =', np.int32(training_set.shape).tolist()
    drange_orig = training_set.get_dynamic_range()
    if verbose: print 'Dynamic range =', drange_orig
    return training_set, drange_orig

def load_dataset_for_previous_run(result_subdir, **kwargs):
    dataset = None
    with open(os.path.join(result_subdir, 'config.txt'), 'rt') as f:
        for line in f:
            if line.startswith('dataset = '):
                exec line
    return load_dataset(dataset, **kwargs)

#----------------------------------------------------------------------------

#主训练函数
def train_gan(
    separate_funcs          = False,
    D_training_repeats      = 1,
    G_learning_rate_max     = 0.0010,
    D_learning_rate_max     = 0.0010,
    G_smoothing             = 0.999,
    adam_beta1              = 0.0,
    adam_beta2              = 0.99,
    adam_epsilon            = 1e-8,
    minibatch_default       = 16,
    minibatch_overrides     = {},
    rampup_kimg             = 40/2,
    rampdown_kimg           = 0,
    lod_initial_resolution  = 4,
    lod_training_kimg       = 400/2,
    lod_transition_kimg     = 400/2,
    #lod_training_kimg       = 40,
    #lod_transition_kimg     = 40,
    total_kimg              = 10000/2,
    dequantize_reals        = False,
    gdrop_beta              = 0.9,
    gdrop_lim               = 0.5,
    gdrop_coef              = 0.2,
    gdrop_exp               = 2.0,
    drange_net              = [-1,1],
    drange_viz              = [-1,1],
    image_grid_size         = None,
    #tick_kimg_default       = 1,
    tick_kimg_default       = 50/5,
    tick_kimg_overrides     = {32:20, 64:10, 128:10, 256:5, 512:2, 1024:1},
    image_snapshot_ticks    = 4,
    network_snapshot_ticks  = 40,
    image_grid_type         = 'default',
    #resume_network_pkl      = '006-celeb128-progressive-growing/network-snapshot-002009.pkl',
    resume_network_pkl      = None,
    resume_kimg             = 0,
    resume_time             = 0.0):

    # Load dataset and build networks.
    training_set, drange_orig = load_dataset() 
    # training_set是dataset模块解析h5之后的对象，
    # drange_orig 为training_set.get_dynamic_range()

    if resume_network_pkl:
        print 'Resuming', resume_network_pkl
        G, D, _ = misc.load_pkl(os.path.join(config.result_dir, resume_network_pkl))
    else:
        G = network.Network(num_channels=training_set.shape[1], resolution=training_set.shape[2], label_size=training_set.labels.shape[1], **config.G)
        D = network.Network(num_channels=training_set.shape[1], resolution=training_set.shape[2], label_size=training_set.labels.shape[1], **config.D)
    Gs = G.create_temporally_smoothed_version(beta=G_smoothing, explicit_updates=True)
    
    # G,D对象可以由misc解析pkl之后生成，也可以由network模块构造

    misc.print_network_topology_info(G.output_layers)
    misc.print_network_topology_info(D.output_layers)

    # Setup snapshot image grid.
    # 设置中途输出图片的格式
    if image_grid_type == 'default':
        if image_grid_size is None:
            w, h = G.output_shape[3], G.output_shape[2]
            image_grid_size = np.clip(1920 / w, 3, 16), np.clip(1080 / h, 2, 16)
        example_real_images, snapshot_fake_labels = training_set.get_random_minibatch(np.prod(image_grid_size), labels=True)
        snapshot_fake_latents = random_latents(np.prod(image_grid_size), G.input_shape)
    else:
        raise ValueError('Invalid image_grid_type', image_grid_type)

    # Theano input variables and compile generation func.
    print 'Setting up Theano...'
    real_images_var  = T.TensorType('float32', [False] * len(D.input_shape))            ('real_images_var')
    # <class 'theano.tensor.var.TensorVariable'>
    # print type(real_images_var),real_images_var
    real_labels_var  = T.TensorType('float32', [False] * len(training_set.labels.shape))('real_labels_var')
    fake_latents_var = T.TensorType('float32', [False] * len(G.input_shape))            ('fake_latents_var')
    fake_labels_var  = T.TensorType('float32', [False] * len(training_set.labels.shape))('fake_labels_var')
    # 带有_var的均为输入张量
    G_lrate = theano.shared(np.float32(0.0))
    D_lrate = theano.shared(np.float32(0.0))
    # share语法就是用来设定默认值的，返回复制的对象
    gen_fn = theano.function([fake_latents_var, fake_labels_var], 
                            Gs.eval_nd(fake_latents_var, fake_labels_var, ignore_unused_inputs=True), 
                            on_unused_input='ignore')
    
    # gen_fn 是一个函数，输入为：[fake_latents_var, fake_labels_var],  
    #                  输出位：Gs.eval_nd(fake_latents_var, fake_labels_var, ignore_unused_inputs=True), 

    
    '''
    def function(inputs, 
                outputs=None, 
                mode=None, 
                updates=None, 
                givens=None, 
                no_default_updates=False, 
                accept_inplace=False, 
                name=None, 
                rebuild_strict=True, 
                allow_input_downcast=None, 
                profile=None, 
                on_unused_input=None)
    '''
    
    #生成函数

    # Misc init.
    #读入当前分辨率
    resolution_log2 = int(np.round(np.log2(G.output_shape[2])))
    #lod 精细度
    initial_lod = max(resolution_log2 - int(np.round(np.log2(lod_initial_resolution))), 0)
    cur_lod = 0.0
    min_lod, max_lod = -1.0, -2.0
    fake_score_avg = 0.0

    # Save example images.
    snapshot_fake_images = gen_fn(snapshot_fake_latents, snapshot_fake_labels)
    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)
    misc.save_image_grid(example_real_images, os.path.join(result_subdir, 'reals.png'), drange=drange_orig, grid_size=image_grid_size)
    misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % 0), drange=drange_viz, grid_size=image_grid_size)

    # Training loop.
    # 这里才是主训练入口
    # 注意在训练过程中不会跳出最外层while循环，因此更换分辨率等操作必然在while循环里

    #现有图片数
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0

    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    tick_train_out = []
    train_start_time = tick_start_time - resume_time
    while cur_nimg < total_kimg * 1000:

        # Calculate current LOD.
        #计算当前精细度
        cur_lod = initial_lod
        if lod_training_kimg or lod_transition_kimg:
            tlod = (cur_nimg / 1000.0) / (lod_training_kimg + lod_transition_kimg)
            cur_lod -= np.floor(tlod)
            if lod_transition_kimg:
                cur_lod -= max(1.0 + (np.fmod(tlod, 1.0) - 1.0) * (lod_training_kimg + lod_transition_kimg) / lod_transition_kimg, 
                                0.0)
            cur_lod = max(cur_lod, 0.0)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        # 当前分辨率
        minibatch_size = minibatch_overrides.get(cur_res, minibatch_default)
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)

        # Update network config.
        # 更新网络结构
        lrate_coef = misc.rampup(cur_nimg / 1000.0, rampup_kimg)
        lrate_coef *= misc.rampdown_linear(cur_nimg / 1000.0, total_kimg, rampdown_kimg)
        G_lrate.set_value(np.float32(lrate_coef * G_learning_rate_max))
        D_lrate.set_value(np.float32(lrate_coef * D_learning_rate_max))
        if hasattr(G, 'cur_lod'): G.cur_lod.set_value(np.float32(cur_lod))
        if hasattr(D, 'cur_lod'): D.cur_lod.set_value(np.float32(cur_lod))

        # Setup training func for current LOD.
        new_min_lod, new_max_lod = int(np.floor(cur_lod)), int(np.ceil(cur_lod))

        print " cur_lod%f\n  min_lod %f\n new_min_lod %f\n max_lod %f\n new_max_lod %f\n"%(cur_lod,min_lod,new_min_lod,max_lod,new_max_lod)


        if min_lod != new_min_lod or max_lod != new_max_lod:
            print 'Compiling training funcs...'
            min_lod, max_lod = new_min_lod, new_max_lod

            # Pre-process reals.
            real_images_expr = real_images_var
            if dequantize_reals:
                rnd = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
                epsilon_noise = rnd.uniform(size=real_images_expr.shape, low=-0.5, high=0.5, dtype='float32')
                real_images_expr = T.cast(real_images_expr, 'float32') + epsilon_noise # match original implementation of Improved Wasserstein
            real_images_expr = misc.adjust_dynamic_range(real_images_expr, drange_orig, drange_net)
            if min_lod > 0: # compensate for shrink_based_on_lod
                real_images_expr = T.extra_ops.repeat(real_images_expr, 2**min_lod, axis=2)
                real_images_expr = T.extra_ops.repeat(real_images_expr, 2**min_lod, axis=3)

            # Optimize loss.
            G_loss, D_loss, real_scores_out, fake_scores_out = evaluate_loss(G, D, min_lod, max_lod, 
                                                                            real_images_expr, real_labels_var, 
                                                                            fake_latents_var, fake_labels_var, 
                                                                            **config.loss)
            G_updates = adam(G_loss, G.trainable_params(), 
                            learning_rate=G_lrate, beta1=adam_beta1, beta2=adam_beta2, 
                            epsilon=adam_epsilon).items()

            D_updates = adam(D_loss, D.trainable_params(), 
                            learning_rate=D_lrate, beta1=adam_beta1, beta2=adam_beta2, 
                            epsilon=adam_epsilon).items()
            
            D_train_fn = theano.function(
                [real_images_var, real_labels_var, fake_latents_var, fake_labels_var],
                [G_loss, D_loss, real_scores_out, fake_scores_out],
                updates=D_updates, on_unused_input='ignore')
            G_train_fn = theano.function(
                [fake_latents_var, fake_labels_var],
                [],
                updates=G_updates+Gs.updates, on_unused_input='ignore')
    
        for idx in xrange(D_training_repeats):
            mb_reals, mb_labels = training_set.get_random_minibatch(minibatch_size, lod=cur_lod, shrink_based_on_lod=True, labels=True)
            mb_train_out = D_train_fn(mb_reals, mb_labels, random_latents(minibatch_size, G.input_shape), random_labels(minibatch_size, training_set))
            cur_nimg += minibatch_size
            tick_train_out.append(mb_train_out)
        G_train_fn(random_latents(minibatch_size, G.input_shape), random_labels(minibatch_size, training_set))

        # Fade in D noise if we're close to becoming unstable
        fake_score_cur = np.clip(np.mean(mb_train_out[1]), 0.0, 1.0)
        fake_score_avg = fake_score_avg * gdrop_beta + fake_score_cur * (1.0 - gdrop_beta)
        gdrop_strength = gdrop_coef * (max(fake_score_avg - gdrop_lim, 0.0) ** gdrop_exp)
        if hasattr(D, 'gdrop_strength'): D.gdrop_strength.set_value(np.float32(gdrop_strength))

        # Perform maintenance operations once per tick.
        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or cur_nimg >= total_kimg * 1000:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            tick_start_time = cur_time
            tick_train_avg = tuple(np.mean(np.concatenate([np.asarray(v).flatten() for v in vals])) for vals in zip(*tick_train_out))
            tick_train_out = []

            # Print progress.
            print 'tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-9.1f sec/kimg %-6.1f Dgdrop %-8.4f Gloss %-8.4f Dloss %-8.4f Dreal %-8.4f Dfake %-8.4f' % (
                (cur_tick, cur_nimg / 1000.0, cur_lod, minibatch_size, misc.format_time(cur_time - train_start_time), tick_time, tick_time / tick_kimg, gdrop_strength) + tick_train_avg)

            # Visualize generated images.
            if cur_tick % image_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                snapshot_fake_images = gen_fn(snapshot_fake_latents, snapshot_fake_labels)
                misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg / 1000)), drange=drange_viz, grid_size=image_grid_size)

            # Save network snapshot every N ticks.
            if cur_tick % network_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg / 1000)))

    # Write final results.
    misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-final.pkl'))
    training_set.close()
    print 'Done.'
    with open(os.path.join(result_subdir, '_training-done.txt'), 'wt'):
        pass

#----------------------------------------------------------------------------

def evaluate_loss(
    G, D, min_lod, max_lod, real_images_in,
    real_labels_in, fake_latents_in, fake_labels_in,
    type            = 'iwass',
    L2_fake_weight  = 0.1,
    iwass_lambda    = 10.0,
    iwass_epsilon   = 0.001,
    iwass_target    = 1.0,
    cond_type       = 'acgan',
    cond_weight     = 1.0,
    cond_tweak_G    = 1.0): # set cond_tweak_G=0.1 to match original improved Wasserstein implementation

    # Helpers.
    def L2(a, b): return 0 if a is None or b is None else Tmean(T.square(a - b))
    def crossent(a, b): return 0 if a is None or b is None else Tmean(lasagne.objectives.categorical_crossentropy(lasagne.nonlinearities.softmax(a), b))
    rnd = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

    # Evaluate generator.
    fake_images_out = G.eval_nd(fake_latents_in, fake_labels_in, min_lod=min_lod, max_lod=max_lod, ignore_unused_inputs=True)

    # Mix reals and fakes through linear crossfade.
    mixing_factors = rnd.uniform((real_images_in.shape[0], 1, 1, 1), dtype='float32')
    mixed_images_out = real_images_in * (1 - mixing_factors) + fake_images_out * mixing_factors

    # Evaluate discriminator.
    real_scores_out,  real_labels_out  = D.eval_nd(real_images_in,   min_lod=min_lod, max_lod=max_lod, expect_num_outputs=2)
    fake_scores_out,  fake_labels_out  = D.eval_nd(fake_images_out,  min_lod=min_lod, max_lod=max_lod, expect_num_outputs=2)
    mixed_scores_out, mixed_labels_out = D.eval_nd(mixed_images_out, min_lod=min_lod, max_lod=max_lod, expect_num_outputs=2)

    if type == 'iwass': # Improved Wasserstein
        mixed_grads = theano.grad(Tsum(mixed_scores_out), mixed_images_out)
        mixed_norms = T.sqrt(Tsum(T.square(mixed_grads), axis=(1,2,3)))
        G_loss = -Tmean(fake_scores_out)
        D_loss = (Tmean(fake_scores_out) - Tmean(real_scores_out)) + Tmean(T.square(mixed_norms - iwass_target)) * iwass_lambda / (iwass_target**2)
        D_loss += L2(real_scores_out, 0) * iwass_epsilon # additional penalty term to keep the scores from drifting too far from zero
        fake_scores_out = fake_scores_out - real_scores_out # reporting tweak
        real_scores_out = T.constant(0) # reporting tweak
    
    return G_loss, D_loss, real_scores_out, fake_scores_out


if __name__ == "__main__":
    #指定随机种子
    np.random.seed(config.random_seed)
    func_params = config.train
    #config.train 为学习率等参数的设置字典
    func_name = func_params['func']
    del func_params['func']
    globals()[func_name](**func_params)
    
    #globals返回的全局变量字典，此行为调用main train func的入口，后面为参数。
    #**代表把func——params当做字典传进去
    #训练顶层函数：train_gan
#----------------------------------------------------------------------------
