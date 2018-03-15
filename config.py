# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
result_dir = 'results'

#----------------------------------------------------------------------------
# Theano config.

theano_flags                                = dict()
theano_flags['device']                      = 'CUDA'            # New gpuarray backend.
#theano_flags['device']                      = 'gpu'             # Old CUDA backend.
theano_flags['force_device']                = 'True'            # Use GPU or die -- never fall back to CPU.
theano_flags['dnn.enabled']                 = 'True'            # Use cuDNN or die -- never fall back to custom convolution ops.
theano_flags['floatX']                      = 'float32'         # Use float32 for everything by default.
theano_flags['warn_float64']                = 'warn'            # Give a warning when float64 ops are used by accident  (not sure if this actually works).
theano_flags['assert_no_cpu_op']            = 'warn'            # Give a warning for CPU ops are used by accident (not sure if this actually works).
theano_flags['allow_gc']                    = 'True'            # Conserve memory. Performance loss is alleviated by subsequent flags.
theano_flags['gpuarray.preallocate']        = '0.80'            # Preallocate % of GPU memory to reduce cost at runtime (new gpuarray backend).
theano_flags['lib.cnmem']                   = '0.80'            # Preallocate % of GPU memory to reduce cost at runtime (old CUDA backend).
theano_flags['dnn.conv.algo_fwd']           = 'small'           # Gives the best performance.
theano_flags['dnn.conv.algo_bwd_filter']    = 'deterministic'   # Necessary for deterministic training runs.
theano_flags['dnn.conv.algo_bwd_data']      = 'deterministic'   # Necessary for deterministic training runs.
theano_flags['nvcc.fastmath']               = 'True'            # Prefer speed over precision.

#----------------------------------------------------------------------------
# Baseline configuration from
# Appendix A.1: "1024x1024 networks used for CelebA-HQ".

run_desc = 'TBD'
random_seed = 1000
dataset = None

train = dict(                               # Training parameters:
    func                    = 'train_gan',  # Main training func.
    separate_funcs          = True,         # Alternate between training generator and discriminator?
    D_training_repeats      = 1,            # n_{critic}
    G_learning_rate_max     = 0.001,        # \alpha
    D_learning_rate_max     = 0.001,        # \alpha
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    adam_beta1              = 0.0,          # \beta_1
    adam_beta2              = 0.99,         # \beta_2
    adam_epsilon            = 1e-8,         # \epsilon
    minibatch_default       = 16,           # Minibatch size for low resolutions.
    minibatch_overrides     = {256:14, 512:6,  1024:3}, # Minibatch sizes for high resolutions.
    rampup_kimg             = 40,           # Learning rate rampup.
    rampdown_kimg           = 0,            # Learning rate rampdown.
    lod_initial_resolution  = 4,            # Network resolution at the beginning.
    lod_training_kimg       = 600,          # Thousands of real images to show before doubling network resolution.
    lod_transition_kimg     = 600,          # Thousands of real images to show when fading in new layers.
    total_kimg              = 15000,        # Thousands of real images to show in total.
    gdrop_coef              = 0.0,          # Do not inject multiplicative Gaussian noise in the discriminator.
)

G = dict(                                   # Generator architecture:
    func                    = 'G_paper',    # Configurable network template.
    fmap_base               = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay              = 1.0,          # log2 of feature map reduction when doubling the resolution.
    fmap_max                = 512,          # Maximum number of feature maps on any resolution.
    latent_size             = 512,          # Dimensionality of the latent vector.
    normalize_latents       = True,         # Normalize latent vector to lie on the unit hypersphere?
    use_wscale              = True,         # Use equalized learning rate?
    use_pixelnorm           = True,         # Use pixelwise normalization?
    use_leakyrelu           = True,         # Use leaky ReLU?
    use_batchnorm           = False,        # Use batch normalization?
    tanh_at_end             = None,         # Use tanh activation for the last layer? If so, how much to scale the output?
)

D = dict(                                   # Discriminator architecture:
    func                    = 'D_paper',    # Configurable network template.
    fmap_base               = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay              = 1.0,          # log2 of feature map reduction when doubling the resolution.
    fmap_max                = 512,          # Maximum number of feature maps on any resolution.
    mbstat_func             = 'Tstdeps',    # Which minibatch statistic to append as an additional feature map?
    mbstat_avg              = 'all',        # Which dimensions to average the statistic over?
    mbdisc_kernels          = None,         # Use minibatch discrimination layer? If so, how many kernels should it have?
    use_wscale              = True,         # Use equalized learning rate?
    use_gdrop               = False,        # Include layers to inject multiplicative Gaussian noise?
    use_layernorm           = False,        # Use layer normalization?
)

loss = dict(                                # Loss function:
    type                    = 'iwass',      # Improved Wasserstein (WGAN-GP).
    iwass_lambda            = 10.0,         # \lambda
    iwass_epsilon           = 0.001,        # \epsilon_{drift}
    iwass_target            = 1.0,          # \alpha
    cond_type               = 'acgan',      # AC-GAN
    cond_weight             = 1.0,          # Weight of the conditioning terms.
)

#----------------------------------------------------------------------------
# Configuration overrides for individual experiments.

# Section 6.1: "Importance of individual contributions in terms of statistical similarity"
if 1:
    datasets = ['celeb128', 'bedroom128']
    configs = ['gulrajani-et-al-2017', 'progressive-growing', 'small-minibatch', 'revised-training-parameters',
               'minibatch-stddev', 'minibatch-discrimination', 'equalized-learning-rate', 'pixelwise-normalization']
    dataset_idx = 0
    config_idx = 1

    run_desc = '%s-%s' % (datasets[dataset_idx], configs[config_idx])
    h5_paths = {'celeb128': 'celeba-128x128.h5', 'bedroom128': 'lsun-bedroom-256x256-full.h5'}
    mirror_augment = {'celeb128': True, 'bedroom128': False}
    dataset = dict(h5_path=h5_paths[datasets[dataset_idx]], max_labels=0, resolution=128, mirror_augment=mirror_augment[datasets[dataset_idx]])

    train.update(lod_training_kimg=800, lod_transition_kimg=800, rampup_kimg=0, total_kimg=10000, minibatch_overrides={})
    G.update(fmap_base=2048)
    D.update(fmap_base=2048)

    if config_idx < 3: # gulrajani-et-al-2017 -- small-minibatch
        train.update(D_training_repeats=5, adam_beta2=0.9, minibatch_default=64, G_smoothing=0.999**20)
        G.update(latent_size=128, normalize_latents=False, use_leakyrelu=False, use_batchnorm=True, tanh_at_end=1.0)
        D.update(use_layernorm=True)
        loss.update(iwass_epsilon=0.0)

    if config_idx == 0: # gulrajani-et-al-2017
        train.update(lod_initial_resolution=128, lod_training_kimg=0, lod_transition_kimg=0)

    if config_idx == 2: # small-minibatch
        train.update(minibatch_default=16, G_smoothing=0.999**5)

    if config_idx < 4 or config_idx == 5: # without minibatch-stddev
        D.update(mbstat_avg=None)

    if config_idx == 5: # minibatch-discrimination
        D.update(mbdisc_kernels=100)

    if config_idx < 6: # without equalized-learning-rate
        train.update(G_learning_rate_max=0.0001, D_learning_rate_max=0.0001)
        G.update(use_wscale=False)
        D.update(use_wscale=False)

    if config_idx < 7: # without pixelwise-normalization
        G.update(use_pixelnorm=False)

#----------------------------------------------------------------------------
# Utilities for analyzing networks.

#train = dict(func='interpolate_latents', run_id=100, snapshot=None, num_frames=5*60*30, filter_frames=45)
#train = dict(func='calc_inception_scores', run_id=170); theano_flags['gpuarray.preallocate'] = '0'
#train = dict(func='calc_sliced_wasserstein_scores', run_id=2328, num_images=16384, nhoods_per_image=128, dir_repeats=4, dirs_per_repeat=128, log='sliced-wasserstein-16k-128-512.txt')
#train = dict(func='calc_mnistrgb_histogram', run_id=200)

if train['func'] != 'train_gan':
    run_desc = '%s_%d' % (train['func'], train['run_id'])

#----------------------------------------------------------------------------
