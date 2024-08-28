import numpy as np
import os, glob, random, time, sys, pickle, glob, h5py
import argparse
import pyarrow.parquet as pq
from resnet import *
import torch
from torch import distributed as dist
# from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, ConcatDataset

from dataset_loader import *
from regression_Models import *
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


    def test(best_epoch, device, WORLD_SIZE, final_model):
        
        if GLOBAL_RANK == 0: 
            print("best_epoch  ", best_epoch)

        model_to_load = final_model
        
        old_checkpoint = torch.load(model_to_load)
        if GLOBAL_RANK == 0: print("---------> Loaded  Model---",model_to_load)
        ddp_model.load_state_dict(old_checkpoint['model_state_dict'])
        ddp_model.eval()
        outputs = []
        targets = []
        loss_avg = 0
        with torch.no_grad():
        ###Drop in your test loop here but note that each GPU will report the test results independently
        ###if you construct the test loop the same way as train/val
            start_time = time.time()
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

            global_rank = dist.get_rank()
            output_dict = {}
            output_dict["m_true"] = np.concatenate(targets)
            output_dict["m_pred"] = np.concatenate(outputs)
            global_rank = dist.get_rank()
            os.makedirs(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_' + timestr+f'_GPUS_{WORLD_SIZE}'+ f'/test_data_epoch_{best_epoch}_M{Mass}/',exist_ok=True)
            with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_'+timestr +f'_GPUS_{WORLD_SIZE}'+ f'/test_data_epoch_{best_epoch}_M{Mass}/'+f'/Inference_data_test_rank_{global_rank}_epoch_{best_epoch}_M{Mass}.pkl', "wb") as outfile:
                pickle.dump(output_dict, outfile, protocol=2)
            with open(os.environ['SCRATCH'] + f'/{args.checkpoint_folder}/{decay}_' + timestr+f'_GPUS_{WORLD_SIZE}' + f'/output_file_globalrank_{global_rank}_epoch_{best_epoch}_M{Mass}.txt', 'a') as w:
                w.write('Model: ' + model_to_load + ' \n')
                # w.write('Acc: %f \n', accuracy)
                w.write('Loss:  ' + f"{loss_avg/(i+1)}"+ ' \n')
                w.write('Numsamples: ' + f"{len(test_loader) // WORLD_SIZE // args.batch_size * WORLD_SIZE * args.batch_size}")
            end_time = time.time()
            if GLOBAL_RANK == 0: 
                print("Time taken for testing :  ", np.round(end_time - start_time, 2))
                print(f"------End testing-----Mass  {Mass}")
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
        timestr_length = len(time.strftime("%Y_%m_%d_%H:%M:%S"))
        timestr_tensor = torch.empty(timestr_length, dtype=torch.uint8).to(device)
    dist.broadcast(timestr_tensor, src=0)
    timestr = timestr_tensor.cpu().numpy().tobytes().decode('utf-8')
    print('timestamp: ' + timestr)

    args.cuda = not args.no_cuda and torch.cuda.is_available()


    

    
    file_test = glob.glob(f'{test_data_path}/*M{Mass}*')[0]

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

    
   
    if GLOBAL_RANK==0:
        
        print("Number of test sets  :  ", n_total_test, "  used test sets:  ", len(test_indices),"---->", (len(test_indices)/n_total_test)*100,"%")


    
    
    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.MSELoss().to(device)

    model = ResNet(len(indices), resblocks, [8,16,32,64])
    # model = ModifiedResNet(resnet_='resnet18',input_channels=len(indices))
    model = model.to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    final_model_dir = args.final_model_dir
    final_model = f"{final_model_dir}/ckpt_{best_epoch}.pt"

        

   
    

    
 
   
   
    if GLOBAL_RANK == 0: print(f'-------------Starting Testing-----Mass {Mass}-----------')
    
    test(best_epoch, device, WORLD_SIZE, final_model)

    sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='', ###Replace with your log dir
                        help='log directory')
    parser.add_argument('--best_epoch', type=int, default=-1,
                        help='best epoch to test model')
    parser.add_argument('--final_model_dir', default='/pscratch/sd/b/bbbam/resnet34_modi_final_Nodes_4.0/13_channels_massregressor_multi_node_2024_07_15_12_30_GPUS_16/Models', 
                        help='final test model')
    parser.add_argument('--test_data_path', default='/pscratch/sd/b/bbbam/IMG_v3_signal_with_trigger_normalized_h5', 
                        help='log directory')
    parser.add_argument('--batch_size', type=int, default=1024, ###With DDP, set this as high as possible
                        help='input batch size for training')

    parser.add_argument('--n_test', type=int, default=-1,
                        help='number of testing sample -1 to use all')
    
    parser.add_argument('--checkpoint_folder', default='ckpts', ###Replace with your ckpts dir (Expected to be in scratch)
                        help='checkpoint file format')
    parser.add_argument('--seed', type=int, default=3333,
                        help='random seed')
    parser.add_argument('--backend', type=str, default="nccl", choices=['nccl', 'gloo'])
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='turn off all cuda devices')
    parser.add_argument('--timestr', type=str, default=None)
    parser.add_argument('--mass', type=str, default='3p7', help="select signal mass from 3p7,4,5,6,8,10,12 and 14")
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--m0_scale', type=float, default=17.2)
    parser.add_argument('-b', '--resblocks',  default=3,     type=int, help='Number of residual blocks.')
    parser.add_argument('-ch','--channels', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12], help='List of channels used')
    parser.add_argument('--mean', type=float, default=9.182514)
    parser.add_argument('--std' , type=float, default=4.5799513)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    best_epoch = args.best_epoch
    m0_scale = args.m0_scale
    Mass = args.mass
    test_data_path = args.test_data_path
    resblocks = args.resblocks
    n_test = args.n_test
    mean_, std_ = args.mean, args.std    
    channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy","HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2" ,"Tob_1", "Tob_2"]
    
    indices = args.channels # channel selected for training change it to class defination too
    channels_used = [channel_list[ch] for ch in indices]
    layers_names = ' | '.join(channels_used)

    decay = f'{len(indices)}_channels_massregressor_multi_node_inference_Mass_{Mass}'



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
    
    main()
    
