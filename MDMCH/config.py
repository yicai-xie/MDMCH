import warnings


class DefaultConfig(object):
    load_img_path = './MDMCH-demo/checkpoints/image_model.pth'  # load model path
    load_txt_path = './MDMCH-demo/checkpoints/text_model.pth'
    pretrain_model_path = './MDMCH-demo/data/imagenet-vgg-f.mat'

    # for FLICKR-25K
    data_path = './FLICKR-25K/FLICKR-25K.mat'
    path_W_sematic_distance = './FLICKR-25K/DCMH-MultiLayers.mat'
    path_S_Multi_value = './FLICKR-25K/S_Multi_value.mat'

    training_size = 10000
    query_size = 2000
    database_size = 18015
    batch_size = 128


    # hyper-parameters
    max_epoch = 10
    alpha = 1
    beta = 1
    eta = 1
    gamma = 1
    miu = 1

    bit = 64
    # final binary code length
    lr = 10 ** (-2)  # initial learning rate
    lr_min = 10**(-6)
    retriving_index = 0

    use_gpu = True

    valid = True

    print_freq = 2  # print info every N epoch

    result_dir = 'result'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
