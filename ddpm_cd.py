import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from model.cd_modules.cd_head import cd_head
from misc.print_diffuse_feats import print_feats

if __name__ == "__main__":
    # ArgumentParser() 新建一个ArgumentParser对象，来获取参数  parser(解析器)  ArgumentParser(参数解析器) add_argument(添加参数)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm_cd.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='1')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    # parser.parse_args() 解析并获得命令行传来的参数
    args = parser.parse_args()
    # print(args) Namespace(config='config/levir_test.json', debug=False, enable_wandb=False, gpu_ids='1', log_eval=True, phase='test')
    # print(type(args)) <class 'argparse.Namespace'>
    # print(vars(args)) {'config': 'config/levir_test.json', 'phase': 'test', 'gpu_ids': '1', 'debug': False, 'enable_wandb': False, 'log_eval': True}
    opt = Logger.parse(args)

    # print(opt) 参数的合集
    # print(type(opt)) <class 'collections.OrderedDict'>

    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt) #设置完了所有参数

    # logging
    # 设置为True，说明设置为使用非确定性算法：
    torch.backends.cudnn.enabled = True
    # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    torch.backends.cudnn.benchmark = True

    # 初始化train.log
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    # 初始化test.log
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)

    # 因为在创建train.log的logger对象时没有指定名字,所以train.log为root logger
    logger = logging.getLogger('base')
    # 控制log格式,打印log进入train.log和screen
    logger.info(Logger.dict2str(opt))
    # 将条目直接写入 log_dir 中的事件文件以供 TensorBoard 使用。不知道什么意思
    tb_logger = SummaryWriter(logdir=opt['path']['tb_logger'])


    # Initialize WandbLogger 初始化 WandbLogger
    if opt['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        # Training log
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        # Validation log
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # Loading change-detction datasets.
    # 载入数据集
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] change-detection dataloader.")
            train_set   = Data.create_cd_dataset(dataset_opt, phase)
            train_loader= Data.create_dataloader(
                train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Creating [val] change-detection dataloader.")
            val_set   = Data.create_cd_dataset(dataset_opt, phase)
            val_loader= Data.create_cd_dataloader(
                val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)
        
        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] change-detection dataloader.")
            print(phase)
            # 设置加载的图片
            test_set   = Data.create_cd_dataset(dataset_opt, phase)
            # 设置torch.utils.data.DataLoader
            test_loader= Data.create_cd_dataloader(
                test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)
    
    logger.info('Initial Dataset Finished')




    # Loading diffusion model
    # 载入diffusion模型
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    # 设置diffusion的噪声
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # Creating change-detection model
    # 创建CD模型
    change_detection = Model.create_CD_model(opt)

    #print("11111111111111",change_detection,"22222222222222222222222")
    
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']  # n_epoch=120
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):         
            change_detection._clear_cache()
            train_result_path = '{}/train/{}'.format(opt['path']
                                                 ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)
            
            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % change_detection.optCD.param_groups[0]['lr']
            logger.info(message)
            for current_step, train_data in enumerate(train_loader):
                # Feeding data to diffusion model and get features
                diffusion.feed_data(train_data)

                f_A=[]
                f_B=[]
                for t in opt['model_cd']['t']:
                    fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                    if opt['model_cd']['feat_type'] == "dec":
                        f_A.append(fd_A_t)
                        f_B.append(fd_B_t)
                        # Uncommet the following line to visualize features from the diffusion model
                        # for level in range(0, len(fd_A_t)):
                        #     print_feats(opt=opt, train_data=train_data, feats_A=fd_A_t, feats_B=fd_B_t, level=level, t=t)
                        # del fe_A_t, fe_B_t
                    else:
                        f_A.append(fe_A_t)
                        f_B.append(fe_B_t)
                        del fd_A_t, fd_B_t
                
                # for i in range(0, len(fd_A)):
                #     print(fd_A[i].shape)

                # Feeding features from the diffusion model to the CD model
                change_detection.feed_data(f_A, f_B, train_data)
                change_detection.optimize_parameters()
                change_detection._collect_running_batch_states()

                # log running batch status
                if current_step % 4 == 0:
                    # message
                    logs = change_detection.get_current_log()
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' %\
                      (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_cd'], logs['running_acc'])
                    logger.info(message)

                    #vissuals
                    visuals = change_detection.get_current_visuals()

                    img_mode = "grid"
                    if img_mode == "single":
                        # Converting to uint8
                        img_A   = Metrics.tensor2img(train_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B   = Metrics.tensor2img(train_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                        #save imgs
                        Metrics.save_img(
                                img_A, '{}/img_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                img_B, '{}/img_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                pred_cm, '{}/img_pred_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                gt_cm, '{}/img_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                        visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                        grid_img = torch.cat((  train_data['A'], 
                                    train_data['B'], 
                                    visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                    visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                    dim = 0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                
            ### log epoch status ###
            change_detection._collect_epoch_states()
            logs = change_detection.get_current_log()
            message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)
            
            if wandb_logger:
                    wandb_logger.log_metrics({
                        'training/mF1': logs['epoch_acc'],
                        'training/mIoU': logs['miou'],
                        'training/OA': logs['acc'],
                        'training/change-F1': logs['F1_1'],
                        'training/no-change-F1': logs['F1_0'],
                        'training/change-IoU': logs['iou_1'],
                        'training/no-change-IoU': logs['iou_0'],
                        'training/train_step': current_epoch
                    })

            change_detection._clear_cache()
            change_detection._update_lr_schedulers()
            
            ##################
            ### validation ###
            ##################
            if current_epoch % opt['train']['val_freq'] == 0:
                val_result_path = '{}/val/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    # Feed data to diffusion model
                    diffusion.feed_data(val_data)
                    f_A=[]
                    f_B=[]
                    for t in opt['model_cd']['t']:
                        fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                        if opt['model_cd']['feat_type'] == "dec":
                            f_A.append(fd_A_t)
                            f_B.append(fd_B_t)
                            del fe_A_t, fe_B_t
                        else:
                            f_A.append(fe_A_t)
                            f_B.append(fe_B_t)
                            del fd_A_t, fd_B_t

                    # Feed data to CD model
                    change_detection.feed_data(f_A, f_B, val_data)
                    change_detection.test()
                    change_detection._collect_running_batch_states()
                    
                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs        = change_detection.get_current_log()
                        message     = '[Validation CD]. epoch: [%d/%d]. Itter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_epoch, n_epoch-1, current_step, len(val_loader), logs['running_acc'])
                        logger.info(message)

                        #vissuals
                        visuals = change_detection.get_current_visuals()

                        img_mode = "grid"
                        if img_mode == "single":
                            # Converting to uint8
                            img_A   = Metrics.tensor2img(val_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                            img_B   = Metrics.tensor2img(val_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                            gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                            pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                            #save imgs
                            Metrics.save_img(
                                img_A, '{}/img_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                img_B, '{}/img_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                pred_cm, '{}/img_pred_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                gt_cm, '{}/img_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        else:
                            # grid img
                            visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                            visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                            grid_img = torch.cat((  val_data['A'], 
                                        val_data['B'], 
                                        visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                            grid_img = Metrics.tensor2img(grid_img)  # uint8
                            Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))

                change_detection._collect_epoch_states()
                logs     = change_detection.get_current_log()
                message = '[Validation CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/mF1': logs['epoch_acc'],
                        'validation/mIoU': logs['miou'],
                        'validation/OA': logs['acc'],
                        'validation/change-F1': logs['F1_1'],
                        'validation/no-change-F1': logs['F1_0'],
                        'validation/change-IoU': logs['iou_1'],
                        'validation/no-change-IoU': logs['iou_0'],
                        'validation/val_step': current_epoch               
                    })
                
                if logs['epoch_acc'] > best_mF1:
                    is_best_model = True
                    best_mF1 = logs['epoch_acc']
                    logger.info('[Validation CD] Best model updated. Saving the models (current + best) and training states.')
                else:
                    is_best_model = False
                    logger.info('[Validation CD]Saving the current cd model and training states.')
                logger.info('--- Proceed To The Next Epoch ----\n \n')

                change_detection.save_network(current_epoch, is_best_model = is_best_model)
                change_detection._clear_cache()

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})
                
        logger.info('End of training.')
    else: # test过程
        logger.info('Begin Model Evaluation (testing).')
        test_result_path = '{}/test/'.format(opt['path']
                                                 ['results'])
        # 在结果的result文件夹内新建装结果文件的test文件夹
        os.makedirs(test_result_path, exist_ok=True)
        # 得到 test logger
        logger_test = logging.getLogger('test')

        # -----------------清除缓存，使一个东西变为未初始化--------------------
        change_detection._clear_cache()

        # enumerate 将一个可遍历的数据对象组合为一个索引序列
        for current_step, test_data in enumerate(test_loader):
            # Feed data to diffusion model
            # 投入数据集------------------------------------------
            diffusion.feed_data(test_data)
            f_A=[]
            f_B=[]
            for t in opt['model_cd']['t']:
                # 获取图片的特征表示
                # np.random.randint(low=2, high=8)-----------
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t)
                if opt['model_cd']['feat_type'] == "dec":
                    f_A.append(fd_A_t)
                    f_B.append(fd_B_t)
                    del fe_A_t, fe_B_t
                else:
                    f_A.append(fe_A_t)
                    f_B.append(fe_B_t)
                    del fd_A_t, fd_B_t

            # -----------把特征给CD模型
            # Feed data to CD model
            change_detection.feed_data(f_A, f_B, test_data)
            change_detection.test() # ----------对数据进行测试
            change_detection._collect_running_batch_states() # ---------当前运行批次的收集状态

            # Logs
            logs        = change_detection.get_current_log()
            message     = '[Testing CD]. Itter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_step, len(test_loader), logs['running_acc'])
            logger_test.info(message)

            # Vissuals 获得变化检测图片并保存
            visuals = change_detection.get_current_visuals()
            img_mode = 'single'
            if img_mode == 'single':
                # Converting to uint8
                visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                img_A   = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                img_B   = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                # Save imgs
                Metrics.save_img(
                    img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))
            else:
                # grid img
                visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                grid_img = torch.cat((  test_data['A'], 
                                        test_data['B'], 
                                        visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                grid_img = Metrics.tensor2img(grid_img)  # uint8
                Metrics.save_img(
                    grid_img, '{}/img_A_B_pred_gt_{}.png'.format(test_result_path, current_step))

        change_detection._collect_epoch_states()
        logs = change_detection.get_current_log()
        message = '[Test CD summary]: Test mF1=%.5f \n' %\
                      (logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
        logger_test.info(message)

        if wandb_logger:
            wandb_logger.log_metrics({
                'test/mF1': logs['epoch_acc'],
                'test/mIoU': logs['miou'],
                'test/OA': logs['acc'],
                'test/change-F1': logs['F1_1'],
                'test/no-change-F1': logs['F1_0'],
                'test/change-IoU': logs['iou_1'],
                'test/no-change-IoU': logs['iou_0'],
            })

        logger.info('End of testing...')
