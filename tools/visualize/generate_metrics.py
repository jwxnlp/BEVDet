# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2023/11/28
#**************************************************************************************
import os, shutil, argparse, glob, tqdm

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Collect GT INFOs for the whole invs Occupancy Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #----------------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--work_dir", type=str, default=r"/mnt/luci-nas/jiangwei/projects/BEVDet/work_dirs/bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal-bevpanv2-aspp3do", help=""
    )
    
    parser.add_argument(
        "--gpu_ids", type=str, default="1,2,3,4,5,6,7", help="GPU ids"
    )
    return parser.parse_args()


# main functions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    #---------------------------------------------------------------------------------
    config_paths = glob.glob(os.path.join(args.work_dir, "bevdet-occ-r50-4dlongterm-stereo-24e_384704*.py"))
    assert len(config_paths) == 1, "ERROR: [ {} ]: Wrong Number of Config Files!".format(len(config_paths))
    config_path = config_paths[0]
    
    ema_pth_paths = glob.glob(os.path.join(args.work_dir, "epoch_*_ema.pth"))
    ema_pth_paths = sorted(ema_pth_paths, 
        key=lambda path: int(os.path.basename(path).split("_")[1]))
    epoch_ids = sorted([int(os.path.basename(path).split("_")[1]) for path in ema_pth_paths],
        reverse=True)
    
    #------------------------------------------------------------------------------------------------------
    gpu_ids = list(map(str, args.gpu_ids.split(",")))
    for epoch_id in tqdm.tqdm(epoch_ids):
        save_pred_path = os.path.join(args.work_dir, "epoch_{}_ema_pred.pkl".format(epoch_id))
        
        if os.path.exists(save_pred_path):
            continue
        
        ema_pth_path = os.path.join(args.work_dir, "epoch_{}_ema.pth".format(epoch_id))
        
        cmd = "CUDA_VISIBLE_DEVICES={} /bin/bash ./tools/dist_test.sh {} {} {} --eval mAP".format(
            args.gpu_ids, config_path, ema_pth_path, len(gpu_ids))

        state = os.system(cmd)
        
        if state != 0:
            print("ERROR: [ {} ]: Not Finished!".format(cmd))
            save_dirs = glob.glob(os.path.join(args.work_dir, "epoch_{}_ema_*".format(epoch_id)))
            for save_dir in save_dirs:
                shutil.rmtree(save_dir)
    
    
    return

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
    main(parse_args())
