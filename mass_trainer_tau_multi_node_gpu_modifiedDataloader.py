import numpy as np
import os, glob, random, time, sys, pickle, glob
import argparse
import pyarrow.parquet as pq
from resnet import *
import torch
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset
import wandb

run_logger = True
hcal_scale  = torch.tensor(0.02)
ecal_scale  = torch.tensor(0.02)
pt_scale    = torch.tensor(0.01)
dz_scale    = torch.tensor(0.0005)
d0_scale    = torch.tensor(0.002)
m0_scale    = torch.tensor(17.2)



def logger(s):
    global f, run_logger
    print(s)
    if run_logger:
        f.write('%s\n'%str(s))

def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs().to(device)
    return loss.mean()



def transform_y(y):
    return y/m0_scale

def inv_transform(y):
    return y*m0_scale

class ParquetDataset(Dataset):
    def __init__(self, filename, label):
        self.parquet = pq.ParquetFile(filename)
        self.cols = ['X_jet.list.item.list.item.list.item','am']
        self.label = label
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jet'] = torch.tensor(data['X_jet'][0], dtype=torch.float32)
        data['am'] = torch.tensor(data['am'], dtype=torch.float32)
        return (data['X_jet'], data['am'])
    def __len__(self):
        return self.parquet.num_row_groups

def zs(data, indices):
    data[:, 0, :, :] = pt_scale * data[:, 0, :, :]  #Track pT
    data[:, 1, :, :] = dz_scale * data[:, 1, :, :] #Track dZ
    data[:, 2, :, :] = d0_scale * data[:, 2, :, :] #Track d0
    data[:, 3, :, :] = ecal_scale * data[:, 3, :, :] #ECAL
    data[:, 4, :, :] = hcal_scale * data[:, 4, :, :] #HCAL
    # Preprocessing
    # High Value Suppression
    data[:, 1, :, :][abs(data[:, 1, :, :]) > 2000 ] = 0
    data[:, 2, :, :][abs(data[:, 2, :, :]) > 1000 ] = 0
    # Zero-Suppression
    data[:, 0, :, :][data[:, 0, :, :] < 1.e-3] = 0.
    data[:, 3, :, :][data[:, 3, :, :] < 1.e-3] = 0.
    data[:, 4, :, :][data[:, 4, :, :] < 1.e-3] = 0.
    data = torch.index_select(data, 1, indices)
    return data


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

    sync_file = _get_sync_file()

    dist.init_process_group(backend=args.backend, world_size=WORLD_SIZE, rank=GLOBAL_RANK, init_method=sync_file)


    args.cuda = not args.no_cuda and torch.cuda.is_available()


    set_random_seeds(random_seed=args.seed)

    ###Drop in your train, validation, and test datasets of type Dataset
    ###The drop_last=True is necessary to ensure each gpu sees the same amount of data

    # train_decays = glob.glob('/pscratch/sd/b/bbbam/DYToTauTau_M-50_13TeV_valid.parquet')
    train_decays = glob.glob ('/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_train/*parquet')
    dset_train = ConcatDataset([ParquetDataset('%s'%d, i) for i, d in enumerate(train_decays)])

    # val_decays = glob.glob('/pscratch/sd/b/bbbam/DYToTauTau_M-50_13TeV_valid.parquet')
    val_decays = glob.glob('/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_valid/*parquet')
    dset_val = ConcatDataset([ParquetDataset('%s'%d, i) for i, d in enumerate(val_decays)])

    # test_decays = glob.glob('/pscratch/sd/b/bbbam/DYToTauTau_M-50_13TeV_valid.parquet')
    test_decays = glob.glob('/pscratch/sd/b/bbbam/signal/IMG_H_AATo4Tau_Hadronic_tauDR0p4_M3p7_signal_v2*parquet')
    dset_test = ConcatDataset([ParquetDataset('%s'%d, i) for i,d in enumerate(test_decays)])


    if GLOBAL_RANK==0:
        print("Number of train set  :  ", len(dset_train))
        print("Number of val set  :  ", len(dset_val))
        print("Number of test set  :  ", len(dset_test))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dset_train, seed=args.seed, num_replicas=WORLD_SIZE, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=args.batch_size,num_workers=16, sampler=train_sampler, drop_last=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dset_val, seed=args.seed, num_replicas=WORLD_SIZE, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dset_val, batch_size=args.batch_size,num_workers=16, sampler=val_sampler)


    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dset_test, seed=args.seed, num_replicas=WORLD_SIZE, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        dset_test, batch_size=args.batch_size,num_workers=16, sampler=test_sampler)

    criterion = nn.BCEWithLogitsLoss().to(device)

    model = resnet34_modified(input_channels=len(indices), num_classes=1)
    model = model.to(device)

    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)


    if args.resume_epoch_num !=0:
        model_to_load = glob.glob(f"{os.environ['SCRATCH']}/{args.checkpoint_folder}/{decay}_*/ckpt_{args.resume_epoch_num}.pt")[0]
        old_checkpoint = torch.load(model_to_load)
        if GLOBAL_RANK == 0: print("---------> Loaded pretrained Model---",model_to_load)
        ddp_model.load_state_dict(old_checkpoint['model_state_dict'])

        # map_location = {"cuda:0": "cuda:{}".format(LOCAL_RANK)}
        # ddp_model.load_state_dict(torch.load(args.resume, map_location=map_location))

    ###Can replace with optimizer of your choice
    optimizer = torch.optim.Adam(ddp_model.parameters(),
                                 lr=args.base_lr*WORLD_SIZE,
                                 eps=args.epsilon,
                                )

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

    ###This part determines the appropriate number of batches per epoch.
    ###Each gpu will see the same amount of data between using this and using drop_last=True in the loaders
    num_batches_per_epoch = len(train_loader.dataset) // WORLD_SIZE // args.batch_size
    scheduler = WarmupScheduler(optimizer, warmup_epochs=args.warmup, total_epochs=args.epochs,
                                num_batches_per_epoch=num_batches_per_epoch)

    def train(epochs, optimizer):
        best_loss = 100
        best_epoch = 0
        for ep in range( epochs):
            epoch = ep + 1 + args.resume_epoch_num
            ddp_model.train()
            if GLOBAL_RANK == 0:
                print('Epoch #', epoch )
            loss_avg = 0
            train_loader.sampler.set_epoch(epoch)
            for i, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    data = zs(data, indices)
                    target = transform_y(target)
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = criterion(output, target)
                loss.backward()
                loss_avg += loss.item()
                optimizer.step()

                if i < num_batches_per_epoch:
                    scheduler.step()
                if (i % 50 == 0) and (GLOBAL_RANK == 0):
                    print(f"Train  :  {i+1}/{len(train_loader)}  Loss:  {loss_avg/(i+1)}")

            if GLOBAL_RANK == 0:

                print('Epoch #:', epoch, 'Avg Train Loss: ', loss_avg/(i+1))
                # wandb.log({"Train_loss": loss_avg/(i+1)})
                ###Add save checkpoints
                os.makedirs(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_' + timestr,exist_ok=True)
                checkpoint_format = os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_' + timestr + f'/ckpt_{epoch}.pt'
                model_dict = {'model_state_dict': ddp_model.state_dict()}
                torch.save(model_dict, checkpoint_format)
                print('---------------Start validation--------------')

            best_epoch, best_loss= validate(epoch, best_epoch, best_loss)
        return best_epoch


    def validate(epoch, best_epoch, best_loss):
        ddp_model.eval()
        loss_avg = 0
        outputs = []
        targets = []
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                data = zs(data, indices)
                target = transform_y(target)
                output = ddp_model(data)
                loss = criterion(output, target)
                loss_avg += loss.item()
                if (i % 50 == 0) and (GLOBAL_RANK == 0):
                    print(f"Validation  :  {i + 1}/{len(val_loader)}  Loss:  {loss_avg/(i+1)}")
                outputs.append(output.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())

        ###Check some condition to determine whether to save the best model
        output_dict = {}
        output_dict["m_true"] = np.concatenate(targets)
        output_dict["m_pred"] = np.concatenate(outputs)
        global_rank = dist.get_rank()
        os.makedirs(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_'+ timestr+f'/valid_data_epoch_{epoch}/',exist_ok=True)
        with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_'+ timestr+f'/valid_data_epoch_{epoch}/'+f'Inference_data_valid_rank_{global_rank}_epoch_{epoch}.pkl', "wb") as outfile:
              pickle.dump(output_dict, outfile, protocol=2)

        # dist.barrier()
        if GLOBAL_RANK == 0:
            print('Epoch #:', epoch, 'Avg Valid Loss: ', loss_avg/(i+1))
            if (loss_avg/(i+1)) < best_loss:
                best_loss = (loss_avg/(i+1))
                best_epoch = epoch
            # wandb.log({"valid_loss": loss_avg/(i+1)})


        return best_epoch, best_loss

    def test(best_epoch_, device, WORLD_SIZE):
        # itos = output_vocab.get_itos()
        model_to_load = glob.glob(f"{os.environ['SCRATCH']}/{args.checkpoint_folder}/{decay}_*/ckpt_{best_epoch_}.pt")[0]
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
            for i, (data,target) in enumerate(test_loader):
                if args.cuda:
                    data, target = data.to(device), target.to(device)

                data = zs(data, indices)
                target = transform_y(target)
                output = ddp_model(data)
                outputs.append(output.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
                loss = criterion(output, target)
                loss_avg += loss.item()

                if (i % 50 == 0) and (GLOBAL_RANK == 0):
                    print(f"Test  :  {i + 1}/{len(test_loader)}  Loss:  {loss_avg/(i+1)}")


            # wandb.log({"valid_loss": loss_avg/(i+1)})

            global_rank = dist.get_rank()
            output_dict = {}
            output_dict["m_true"] = np.concatenate(targets)
            output_dict["m_pred"] = np.concatenate(outputs)
            global_rank = dist.get_rank()
            os.makedirs(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_' + timestr + f'/test_data_epoch_{best_epoch}/',exist_ok=True)
            with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_'+timestr+ f'/test_data_epoch_{best_epoch}/'+f'/Inference_data_test_rank_{global_rank}_epoch_{best_epoch_}.pkl', "wb") as outfile:
                pickle.dump(output_dict, outfile, protocol=2)
            with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_' + timestr + f'/output_file_globalrank_{global_rank}_epoch_{best_epoch_}.txt', 'a') as w:
                w.write('Model: ' + model_to_load + ' \n')
                # w.write('Acc: %f \n', accuracy)
                w.write('Loss:  ' + f"{loss_avg/(i+1)}"+ ' \n')
                w.write('Numsamples: ' + f"{len(test_loader.dataset) // WORLD_SIZE // args.batch_size * WORLD_SIZE * args.batch_size}")
        print("------End testing-----")


    if GLOBAL_RANK == 0: print('---------------------------------Start Training----------------------------')
    ###Drop in your loss function

    ###Get your best epoch after training so you can use it to load the appropriate model for testing
    best_epoch = train(args.epochs, optimizer)
    global_rank = dist.get_rank()
    print(f"best_epoch before synchronization global_rank--{global_rank}------------",best_epoch)
    best_epoch_for_test = best_epoch
    if global_rank == 0:
        print('-------------Starting Testing----------------')
        # dist.broadcast(torch.tensor(best_epoch_for_test, dtype=torch.int).to(device), src=0)
        # print(f"----before barrier. Global rank: {global_rank}")
        # dist.barrier()
        # print(f"----after barrier. Global rank: {global_rank}")
        # print(f"best_epoch_for_test-------global_rank--{global_rank}-------",best_epoch_for_test)
        test(best_epoch_for_test, device, WORLD_SIZE)

    sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='', ###Replace with your log dir
                        help='log directory')
    parser.add_argument('--batch_size', type=int, default=3072, ###With DDP, set this as high as possible
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=90, ###Tune to your problem
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
    parser.add_argument('-b', '--resblocks',  default=2,     type=int, help='Number of residual blocks.')
    parser.add_argument('-ch','--channels', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12], help='List of channels used')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy",
    "HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2",
    "Tib_3", "Tib_4", "Tob_1", "Tob_2", "Tob_3", "Tob_4", "Tob_5",
    "Tob_6", "Tid_1", "Tec_1", "Tec_2", "Tec_3"]

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

    if GLOBAL_RANK==0:
        print(f"Total channel in this training # {len(indices)} :-> ",layers_names)
    if args.timestr is None:
        timestr = time.strftime("%Y_%m_%d_%H_%M")###Get the current time to identify different models
    else:
        timestr = args.timestr
    print('timestamp: ' + timestr)



    # wandb.login(key="51b58a76963008d6010f73edbd6d0617a772c9df")
    # wandb.init(
    #     project = "Boosted Tau classifier",
    #     name = "ResNet_based_Model"
    # )
    main()
    # wandb.finish()
