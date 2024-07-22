import numpy as np
import os, glob, random, time, sys, pickle, glob, h5py
import argparse
import pyarrow.parquet as pq
# from resnet import *
from regression_Models import *
import torch
from torch import distributed as dist
# from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, ConcatDataset
import wandb
from dataset_loader import *
from grokfast import gradfilter_ma, gradfilter_ema
alpha = 0.98
lamb = 2.0
run_logger = True

def logger(s):
    global f, run_logger
    print(s)
    if run_logger:
        f.write('%s\n'%str(s))

def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs().to(device)
    return loss.mean()


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():

    def _get_sync_file():
        """Logic for naming sync file using slurm env variables"""
        sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
        os.makedirs(sync_file_dir,exist_ok=True)
        sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        return sync_file

    

    ###The more gpus you scale to, the less stable the training unless you use this type of scheduler.
    ###It first, increases the learning rate linearly then decreases it like a decaying exponential.
    ###This type of scheduler has been shown to give good performance for multi-node-multi-gpu.
    class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_epochs, total_epochs, num_batches_per_epoch, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.num_batches_per_epoch = num_batches_per_epoch
            super(WarmupScheduler, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch+1 < self.warmup_epochs * self.num_batches_per_epoch:
                return [base_lr * (self.last_epoch+1) / (self.warmup_epochs * self.num_batches_per_epoch) for base_lr in self.base_lrs]
            else:
                lr = [base_lr * (1 - (self.last_epoch+1 - self.warmup_epochs * self.num_batches_per_epoch) / \
                    ((self.total_epochs - self.warmup_epochs) * self.num_batches_per_epoch)) for base_lr in self.base_lrs]
                return lr

   
    def train(epochs, optimizer, WandB_):
        best_loss = 100
        best_epoch = 0
        grads = None
        for ep in range( epochs):
            epoch = ep + 1 + args.resume_epoch_num
            ddp_model.train()
            if GLOBAL_RANK == 0:
                print('Epoch #', epoch )
            loss_avg = 0
            for i, sample in enumerate(train_loader):
                if args.cuda:
                    data, target = sample[0].to(device), sample[1].to(device)
                with torch.no_grad():
                    target = transform_norm_y(target, mass_mean, mass_std)
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = criterion(output, target)
                loss.backward()
                grads = gradfilter_ema(ddp_model, grads=grads, alpha=alpha, lamb=lamb)
                loss_avg += loss.item()
                optimizer.step()

                if i < num_batches_per_epoch:
                    scheduler.step()
                if (i % 50 == 0) and (GLOBAL_RANK == 0):
                    print(f"{epoch} Train  :  {i+1}/{len(train_loader)}  Loss:  {loss_avg/(i+1)}")
                    if WandB_: wandb.log({"Train_loss": loss_avg/(i+1)})
            if GLOBAL_RANK == 0:

                print('Epoch #:', epoch, 'Avg Train Loss: ', loss_avg/(i+1))
                ###Add save checkpoints
                os.makedirs(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_' + timestr+f'_GPUS_{WORLD_SIZE}'+'/Models',exist_ok=True)
                checkpoint_format = os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_' + timestr+ f'_GPUS_{WORLD_SIZE}'+'/Models'+ f'/ckpt_{epoch}.pt'
                model_dict = {'model_state_dict': ddp_model.state_dict()}
                torch.save(model_dict, checkpoint_format)
                print('---------------Start validation--------------')

            best_epoch, best_loss= validate(epoch, best_epoch, best_loss, WandB_)
        return best_epoch


    def validate(epoch, best_epoch, best_loss, WandB_):
        ddp_model.eval()
        loss_avg = 0
        outputs = []
        targets = []
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                if args.cuda:
                    data, target = sample[0].to(device), sample[1].to(device)
                target = transform_norm_y(target, mass_mean, mass_std)
                output = ddp_model(data)
                loss = criterion(output, target)
                loss_avg += loss.item()
                if (i % 50 == 0) and (GLOBAL_RANK == 0):
                    print(f"{epoch} Validation  :  {i + 1}/{len(val_loader)}  Loss:  {loss_avg/(i+1)}")
                    if WandB_: wandb.log({"valid_loss": loss_avg/(i+1)})
                outputs.append(inv_transform_norm_y(output, mass_mean, mass_std).detach().cpu().numpy())
                targets.append(inv_transform_norm_y(target, mass_mean, mass_std).detach().cpu().numpy())

        # Collect loss across all devices
        total_loss_tensor = torch.tensor([loss_avg], dtype=torch.float32, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        loss_avg = total_loss_tensor.item() / WORLD_SIZE

        ###Check some condition to determine whether to save the best model
        output_dict = {}
        output_dict["m_true"] = np.concatenate(targets)
        output_dict["m_pred"] = np.concatenate(outputs)
        global_rank = dist.get_rank()
        os.makedirs(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_'+ timestr+f'_GPUS_{WORLD_SIZE}'+f'/valid_data_epoch_{epoch}/',exist_ok=True)
        with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_'+ timestr+f'_GPUS_{WORLD_SIZE}'+f'/valid_data_epoch_{epoch}/'+f'Inference_data_valid_rank_{global_rank}_epoch_{epoch}.pkl', "wb") as outfile:
              pickle.dump(output_dict, outfile, protocol=2)

        # dist.barrier()
        if GLOBAL_RANK == 0:
            print('Epoch #:', epoch, 'Avg Valid Loss: ', loss_avg/(i+1))
        if (loss_avg/(i+1)) < best_loss:
            best_loss = (loss_avg/(i+1))
            best_epoch = epoch



        return best_epoch, best_loss

    def test(best_epoch, device, WORLD_SIZE, WandB_):
        if GLOBAL_RANK == 0: 
            print("best_epoch  ", best_epoch)

        model_to_load = glob.glob(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_' + timestr+ f'_GPUS_{WORLD_SIZE}'+'/Models'+ f'/ckpt_{best_epoch}.pt')[0]
        
        old_checkpoint = torch.load(model_to_load)
        if GLOBAL_RANK == 0: print("---------> Loaded Best Model---",model_to_load)
        ddp_model.load_state_dict(old_checkpoint['model_state_dict'])
        ddp_model.eval()
        outputs = []
        targets = []
        loss_avg = 0
        with torch.no_grad():
        ###Drop in your test loop here but note that each GPU will report the test results independently
        ###if you construct the test loop the same way as train/val
            for i, sample in enumerate(test_loader):
                if args.cuda:
                    data, target = sample[0].to(device), sample[1].to(device)
                target = transform_norm_y(target, mass_mean, mass_std)
                output = ddp_model(data)
                loss = criterion(output, target)
                loss_avg += loss.item()
                outputs.append(inv_transform_norm_y(output, mass_mean, mass_std).detach().cpu().numpy())
                targets.append(inv_transform_norm_y(target, mass_mean, mass_std).detach().cpu().numpy())



                if (i % 50 == 0) and (GLOBAL_RANK == 0):
                    print(f"Test  :  {i + 1}/{len(test_loader)}  Loss:  {loss_avg/(i+1)}")
                    if WandB_ : wandb.log({"test_loss": loss_avg/(i+1)})

            global_rank = dist.get_rank()
            output_dict = {}
            output_dict["m_true"] = np.concatenate(targets)
            output_dict["m_pred"] = np.concatenate(outputs)
            global_rank = dist.get_rank()
            os.makedirs(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_' + timestr+f'_GPUS_{WORLD_SIZE}'+ f'/test_data_epoch_{best_epoch}/',exist_ok=True)
            with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_'+timestr +f'_GPUS_{WORLD_SIZE}'+ f'/test_data_epoch_{best_epoch}/'+f'/Inference_data_test_rank_{global_rank}_epoch_{best_epoch}_M3p7.pkl', "wb") as outfile:
                pickle.dump(output_dict, outfile, protocol=2)
            with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_' + timestr+f'_GPUS_{WORLD_SIZE}' + f'/output_file_globalrank_{global_rank}_epoch_{best_epoch}.txt', 'a') as w:
                w.write('Model: ' + model_to_load + ' \n')
                # w.write('Acc: %f \n', accuracy)
                w.write('Loss:  ' + f"{loss_avg/(i+1)}"+ ' \n')
                w.write('Numsamples: ' + f"{len(test_loader) // WORLD_SIZE // args.batch_size * WORLD_SIZE * args.batch_size}")
            if GLOBAL_RANK == 0: print("------End testing-----")

    sync_file = _get_sync_file()

    dist.init_process_group(backend=args.backend, world_size=WORLD_SIZE, rank=GLOBAL_RANK, init_method=sync_file)

    if GLOBAL_RANK==0:
        print(f"Total channel in this training # {len(indices)} :-> ",layers_names)
        if args.timestr is None:
            timestr = time.strftime("%Y_%m_%d_%H:%M:%S")###Get the current time to identify different models
        else:
            timestr = args.timestr

        timestr_tensor = torch.tensor(list(timestr.encode('utf-8')), dtype=torch.uint8).to(device)
    else:
        timestr_tensor = torch.empty(len(time.strftime("%Y_%m_%d_%H:%M:%S")), dtype=torch.uint8).to(device)
    dist.broadcast(timestr_tensor, src=0)
    timestr = timestr_tensor.cpu().numpy().tobytes().decode('utf-8')
    print('timestamp: ' + timestr)

    args.cuda = not args.no_cuda and torch.cuda.is_available()


    set_random_seeds(random_seed=args.seed)

    ###Drop in your train, validation, and test datasets of type Dataset
    ###The drop_last=True is necessary to ensure each gpu sees the same amount of data

    file_train = glob.glob(f'{data_path}/*train*')[0]
    file_valid = glob.glob(f'{data_path}/*valid*')[0]
   

    train_dset = RegressionDataset(file_train, preload_size=BATCH_SIZE)
    valid_dset = RegressionDataset(file_valid, preload_size=BATCH_SIZE)
    
    n_total_train = len(train_dset)
    n_total_valid = len(valid_dset)
    

    if n_train != -1:
        train_indices = list(range(n_train))
        random.shuffle(train_indices)
    else:
        train_indices = list(range(n_total_train))
        random.shuffle(train_indices)

    if n_valid !=-1:
        valid_indices = list(range(n_valid))
        random.shuffle(valid_indices)
    else:
        valid_indices = list(range(n_total_valid))
        random.shuffle(valid_indices)

    
    

    train_sampler = ChunkedDistributedSampler(train_indices, chunk_size=BATCH_SIZE, shuffle=True, num_replicas=None, rank=None)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=args.num_workers, drop_last=True)

    val_sampler = ChunkedDistributedSampler(valid_indices, chunk_size=BATCH_SIZE, shuffle=False, num_replicas=None, rank=None)
    val_loader = DataLoader(valid_dset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=args.num_workers, drop_last=True)
   
    if GLOBAL_RANK==0:
        print("Number of train sets  :  ", n_total_train, "   used train sets:  ", len(train_indices),"---->", (len(train_indices)/n_total_train)*100,"%")
        print("Number of val sets    :  ", n_total_valid, "   used valid sets:  ", len(valid_indices),"---->", (len(valid_indices)/n_total_valid)*100,"%")
        
    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.MSELoss().to(device)

    # model = resnet34_modified(input_channels=len(indices), num_classes=1)
    # model = ModifiedResNet(input_channels=len(indices), resnet_='resnet18')
    # model = EfficientNet(in_channels=len(indices), effnet=0)
    model = resnet_all(in_channels=len(indices), resnetX='resnet18')
    # model = CustomCoAtNet(in_channels=len(indices), coatnet='coatnet_0_224')
    model = model.to(device)

    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)


    if args.resume_epoch_num !=0:
        model_to_load = glob.glob(f"{os.environ['SCRATCH']}/{args.checkpoint_folder}_Nodes_{WORLD_SIZE/4}/{decay}_*/Models/ckpt_{args.resume_epoch_num}.pt")[0]
        old_checkpoint = torch.load(model_to_load)
        if GLOBAL_RANK == 0: print("---------> Loaded pretrained Model---",model_to_load)
        ddp_model.load_state_dict(old_checkpoint['model_state_dict'])


    ###Can replace with optimizer of your choice
    optimizer = torch.optim.Adam(ddp_model.parameters(),
                                 lr=args.base_lr*WORLD_SIZE,
                                 eps=args.epsilon,
                                )
    
    ###This part determines the appropriate number of batches per epoch.
    ###Each gpu will see the same amount of data between using this and using drop_last=True in the loaders
    num_batches_per_epoch = len(train_loader.dataset) // WORLD_SIZE // args.batch_size
    scheduler = WarmupScheduler(optimizer, warmup_epochs=args.warmup, total_epochs=args.epochs,
                                num_batches_per_epoch=num_batches_per_epoch)
 
    if GLOBAL_RANK == 0: print('---------------------------------Start Training----------------------------')
    ###Drop in your loss function

    ###Get your best epoch after training so you can use it to load the appropriate model for testing
    best_epoch = train(args.epochs, optimizer, WandB_ )
    # global_rank = dist.get_rank()
    if GLOBAL_RANK == 0: 
        print(f"best_epoch before synchronization global_rank--{GLOBAL_RANK}------------",best_epoch)
        best_epoch_tensor = torch.tensor(best_epoch, dtype=torch.int).to(device)
    else:
        best_epoch_tensor = best_epoch = torch.tensor(0, dtype=torch.int).to(device)

    dist.broadcast(best_epoch_tensor, src=0)
    best_epoch = best_epoch_tensor.item()

    if run_test:
        if GLOBAL_RANK == 0: print('-------------Starting Testing----------------')
        file_test = glob.glob(f'{test_data_path}/*M3p7*')[0]
        test_dset = RegressionDataset(file_test, preload_size=BATCH_SIZE)
        n_total_test = len(test_dset)
        if n_test !=-1:
            test_indices = list(range(n_test))
            random.shuffle(test_indices)
        else:
            test_indices = list(range(n_total_test))
            random.shuffle(test_indices)

        test_sampler = ChunkedDistributedSampler(test_indices, chunk_size=BATCH_SIZE, shuffle=False, num_replicas=None, rank=None)
        test_loader = DataLoader(test_dset, batch_size=BATCH_SIZE, sampler=test_sampler, pin_memory=True, num_workers=args.num_workers, drop_last=True)
        if GLOBAL_RANK==0: print("Number of test sets  :  ", n_total_test, "  used test sets:  ", len(test_indices),"---->", (len(test_indices)/n_total_test)*100,"%")
        
        test(best_epoch, device, WORLD_SIZE, WandB_)

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='', ###Replace with your log dir
                        help='log directory')
    parser.add_argument('--data_path', default='/pscratch/sd/b/bbbam/normalized_nan_replaced_m1p2To17p2_massreg_samples_chunksize_32_h5', 
                        help='log directory')
    parser.add_argument('--test_data_path', default='/pscratch/sd/b/bbbam/signal_with_trigger_normalized_h5', 
                        help='log directory')
    parser.add_argument('--batch_size', type=int, default=1024, ###With DDP, set this as high as possible
                        help='input batch size for training')
    parser.add_argument('--n_train', type=int, default=-1,
                        help='number of training sample -1 to use all')
    parser.add_argument('--n_valid', type=int, default=-1,
                        help='number of validation sample -1 to use all')
    parser.add_argument('--n_test', type=int, default=-1,
                        help='number of testing sample -1 to use all')
    parser.add_argument('--epochs', type=int, default=100, ###Tune to your problem
                        help='number of epochs to train')
    parser.add_argument('--base_lr', type=float, default=0.001, ###Tune to your problem
                        help='learning rate for a single GPU')
    parser.add_argument('--epsilon', type=float, default=0.0000001, ###Tune to your problem
                        help='ADAM epsilon')
    parser.add_argument('--checkpoint_folder', default='ckpts', ###Replace with your ckpts dir (Expected to be in scratch)
                        help='checkpoint file format')
    parser.add_argument('--seed', type=int, default=3333,
                        help='random seed')
    parser.add_argument('--backend', type=str, default="nccl", choices=['nccl', 'gloo'])
    parser.add_argument('--resume', type=str, default=None, ###If you want to resume training from a specific model
                        help='model path to resume from')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='turn off all cuda devices')
    parser.add_argument('--warmup', type=int, default=3, ###This should be ~5-10% of total epochs
                        help='number of warmup epochs before reaching base_lr')
    parser.add_argument('--timestr', type=str, default=None)
    parser.add_argument('--resume_epoch_num', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--m0_scale', type=float, default=17.2)
    parser.add_argument('-b', '--resblocks',  default=2,     type=int, help='Number of residual blocks.')
    parser.add_argument('-ch','--channels', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12], help='List of channels used')
    parser.add_argument('--WandB', action='store_true', help='flag for wandb')
    parser.add_argument('--run_test', action='store_true', help='flag for running test on signal samples')
    parser.add_argument('--mean', type=float, default=8.893934)
    parser.add_argument('--std' , type=float, default=2.7809753)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    n_train = args.n_train
    n_valid = args.n_valid
    n_test = args.n_test
    m0_scale = args.m0_scale
    data_path = args.data_path
    test_data_path = args.test_data_path
    WandB_ = args.WandB
    run_test = args.run_test
    mean_, std_ = args.mean, args.std    
    channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy","HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2" ,"Tob_1", "Tob_2"]
    
    indices = args.channels # channel selected for training change it to class defination too
    channels_used = [channel_list[ch] for ch in indices]
    layers_names = ' | '.join(channels_used)

    decay = f'{len(indices)}_channels_massregressor_multi_node'



    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    WORLD_SIZE = int(os.environ['SLURM_NTASKS'])  # number of nodes
    GLOBAL_RANK = int(os.environ['SLURM_PROCID'])
    print('num devices: ', torch.cuda.device_count())
    print('global rank: ', GLOBAL_RANK)
    print('local rank:  ', LOCAL_RANK)
    print('cuda device: ', torch.cuda.current_device())

    try:
        device = torch.device("cuda:{}".format(torch.cuda.current_device()))
        indices = torch.tensor(indices).to(device)
    except:
        print('GPU #', LOCAL_RANK, ' is down')
        sys.exit()

    

    m0_scale = torch.tensor(m0_scale)
    mass_mean = torch.tensor(mean_)
    mass_std= torch.tensor(std_)


    
    if WandB_ and GLOBAL_RANK == 0:
        # Use you own key otherwise it mess mine 
        wandb.login(key="51b58a76963008d6010f73edbd6d0617a772c9df")
        wandb.init(
            project = f"Mass Regression using  {WORLD_SIZE/4} nodes",
            name = f"resnet34_modified_gpu_{WORLD_SIZE}"
        )
    main()
    if WandB_ and GLOBAL_RANK == 0: wandb.finish()
    sys.exit()