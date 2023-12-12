# -*- coding: utf-8 -*-
#**************************************************************************************
import os, argparse, json, glob, tqdm, shutil, cv2, random, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import defaultdict

#-----------------------------------------------------------------------------------------------
def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Parse Annotations of DMS Data for Object Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--home_work_dir", type=str, default=r"./work_dirs", help=""
    )
    
    parser.add_argument(
        "--figure_dir", type=str, default=r"./tools/visualize/metric_contrast_figures", help="dir of visualization"
    )
    
    return parser.parse_args()


def parse_metrics(work_dir):
    """"""
    # print("\n" + "+" * 70)
    # print("--- parse metrics from {}".format(work_dir))
    model_name = os.path.basename(work_dir)
    
    ema_pth_paths = glob.glob(os.path.join(work_dir, "epoch_*_ema_eval.pkl"))
    epoch_ids = sorted([int(os.path.basename(path).split("_")[1]) for path in ema_pth_paths],
        reverse=True)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    metrics = dict()
    for epoch_id in epoch_ids:
        metric_pkl_path = os.path.join(work_dir, "epoch_{}_ema_eval.pkl".format(epoch_id))
        with open(metric_pkl_path, "rb") as f:
            metric = pickle.load(f)
        metrics[epoch_id] = metric
    return metrics


def main(args):
    """Main Function"""
    work_names = [
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-original",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effec-100-avg-weight",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.0",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e3",
        "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice",
        "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-resegdice-rescal",
        "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal-bev-aspp3d",
        "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal-bevpan-aspp3d",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-nofree-scal-nofree",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal-bevdbfpn-aspp3do",
        
        "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal-bevdbfpn-aspp3do",
        # "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-1.0-scal-1.0-bev-aspp3do",
        "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal-bevpanv2-aspp3do",
        "bevdet-occ-r50-4dlongterm-stereo-24e_384704-loss-beta-0.9-effc-1e4-mmseg-dice-scal-bevUnetDecoder-aspp3do"

    ]  
    
    work_dirs = [
        os.path.join(args.home_work_dir, work_name) 
        for work_name in work_names if os.path.exists(os.path.join(args.home_work_dir, work_name))]
    metricss = [
        parse_metrics(work_dir) for work_dir in work_dirs]
    if len(metricss) <= 0:
        raise Exception("ERROR: [ {} ]: No Valid Work Dirs!".format(args.home_work_dir))
    
    # visualize
    #----------------------------------------------------------------------------
    if os.path.exists(args.figure_dir):
        shutil.rmtree(args.figure_dir)
    os.makedirs(args.figure_dir)
    
    colors = np.array([
        (255, 0, 0),
        (255, 165, 0),
        (0, 255, 0),
        (0, 255, 255),  
        (0, 0, 255), 
        (139, 0, 255),
        (0, 0, 0)], dtype=np.float) / 255
    fig, ax = plt.subplots()
    for i, (work_dir, metrics) in enumerate(zip(work_dirs, metricss)):
        print("\n" + "+" * 70)
        print("--- parse metrics from {}".format(work_dir))
    
        work_name = os.path.basename(work_dir)
        index = work_name.index("384704")
        label_name = work_name[index+7:]
        
        epoch_ids, mious = [], []
        for epoch_id, metric in metrics.items():
            miou = metric[1][:-1].mean() * 100
            mious.append(miou)
            epoch_ids.append(epoch_id)
        
        epoch_ids = np.array(epoch_ids)
        mious = np.array(mious)
        index = np.argmax(mious)
        print("--- max miou: {} at {} epoch".format(mious[index], epoch_ids[index]))

        ax.plot(
            epoch_ids, mious, 
            color=colors[i], marker=".", markersize=4, 
            linestyle='-', linewidth=1, label=label_name)
        
    # ax.set_title("Loss: Train vs. Val")
    ax.set_title("mIoU")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.legend(loc="best")
    plt.savefig(os.path.join(args.figure_dir, "miou.png"))
    return
    
    



#==================================================================
if __name__ == "__main__":
    main(parse_args())