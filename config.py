class Config():

    batchsize = 32
    num_workers = 8
    lr = 0.001
    num_epochs = 40

    input_size_h = 512
    input_size_w = 512
    n_classes = 4

    seed = 100

    data = '/work/data'
    result = '/work/result/test15'

    feature_extractor = 'as_ef-b0'
    normalize = 'global_norm'
    batch_uniform = False
    metric_learning = False
    fp16 = True
