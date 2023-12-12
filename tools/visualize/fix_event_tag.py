# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2023/11/28
#**************************************************************************************
import os, shutil, argparse

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Collect GT INFOs for the whole invs Occupancy Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #----------------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--src_log_dir", type=str, default=r"/mnt/luci-nas/jiangwei/projects/BEVDet/work_dirs/bevdet-occ-r50-4dlongterm-stereo-24e_384704-noloadfrom/tf_logs", help=""
    )
    parser.add_argument(
        "--target_log_dir", type=str, default=r"/mnt/luci-nas/jiangwei/projects/BEVDet/work_dirs/bevdet-occ-r50-4dlongterm-stereo-24e_384704-noloadfrom/tf_logs_ce", help=""
    )
    
    return parser.parse_args()


# main functions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    if not os.path.exists(args.src_log_dir):
        raise Exception("ERROR: [ {} ]: Not Exist!".format(args.src_log_dir))
    
    # load
    #-----------------------------------------------------------------
    ea = event_accumulator.EventAccumulator(args.src_log_dir)
    ea.Reload()
    
    tags = ea.scalars.Keys()
    
    print(tags)
    
    # write
    #------------------------------------------------------------------------
    if os.path.exists(args.target_log_dir):
        shutil.rmtree(args.target_log_dir)
    os.makedirs(args.target_log_dir)
    writer = SummaryWriter(args.target_log_dir)
    for tag in tags:
        scalar_list = ea.scalars.Items(tag)
        if tag == 'train/loss_occ':
            tag = 'train/loss_ce'
        for scalar in scalar_list:
            writer.add_scalar(tag, scalar.value, scalar.step, scalar.wall_time)
    writer.close()
    return

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
    main(parse_args())
