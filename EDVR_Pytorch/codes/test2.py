from data import create_dataloader, create_dataset


def test():
    dataset_opt = dict()
    dataset_opt['mode'] = 'REDS'
    dataset_opt['interval_list'] = [1]
    dataset_opt['random_reverse'] = False
    dataset_opt['N_frames'] = 5
    dataset_opt['dataroot_GT'] = './../datasets/train_gt_wval.lmdb'
    dataset_opt['dataroot_LQ'] = './../datasets/train_input_wval.lmdb'
    dataset_opt['GT_size'] = 256
    dataset_opt['LQ_size'] = 256
    dataset_opt['scale'] = 4
    dataset_opt['border_mode'] = False
    dataset_opt['cache_keys'] = 'meta_info.pkl'
    dataset_opt['name'] = 'REDS'
    dataset_opt['n_workers'] = 3
    dataset_opt['batch_size'] = 16


    #not find  yml
    dataset_opt['data_type'] = 'lmdb'


    dataset_opt['phase'] = 'train'


    dataset_opt['debug'] = True
    train_set = create_dataset(dataset_opt)
    print(train_set)
    train_sampler = None
    opt = dict()
    opt['dist'] = True
    opt['gpu_ids'] = [0]
    train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
    print(train_loader)

    for _, train_data in enumerate(train_loader):
        print(train_data)
        break

if __name__ == '__main__':
    test()