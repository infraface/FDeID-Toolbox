# -*- coding: utf-8 -*-
# ***********************************************
# Configuration for the both training and testing
# ***********************************************
import os

class Configuration():
    '''
    Models and data initialization
    '''
    checkpoint_dir = ''
    # Pointing to the LADN dataset
    data_dir = '/path/to/datasets/LADN' 
    model_dir = './models_meta_tar_adv' # Save dir for Adv-Makeup checkpoints
    after_dir = 'after'
    before_dir = 'before'
    targets_dir = '/path/to/Adv-Makeup/Datasets_Makeup/target_aligned_600'
    lmk_name = 'landmark.pk' # Landmarks for un- and real-world makeup faces
    use_se = True
    pretrained = False
    
    # Updated model list with user provided models
    # We use AdaFace and ArcFace for meta-training, and CosFace for meta-testing
    train_model_name_list = ['adaface', 'arcface', 'cosface'] 
    val_model_name_list = ['cosface'] # Use CosFace for validation/monitoring

    # Model Paths
    adaface_path = './weight/adaface_pre_trained/adaface_ir50_ms1mv2.ckpt'
    arcface_path = './weight/ms1mv3_arcface_r100_fp16/backbone.pth'
    cosface_path = './weight/glint360k_cosface_r50_fp16_0.1/backbone.pth'
    vgg_path = './weight/advmakeup_pre_trained/vgg16.pth'

    '''
    Params for the model training and testing
    '''
    lr = 0.001 # Learning rate
    update_lr_m = 0.001 # Learning rate for meta-optimization
    epoch_steps = 300
    gpu = 0
    n_threads = 8
    batch_size = 4
    input_dim = 3

    '''
    Input data preprocessing
    '''
    resize_size = (420, 160) # Size of input eye-area
    # Idxes of the eye-area in the facial landmark list
    eye_area = [9, 10, 11, 19, 84, 29, 79, 28, 24, 73, 70, 75, 74, 13, 15, 14, 22]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
