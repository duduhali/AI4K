https://www.kesci.com/home/competition/forum/5dc2c8072ada7a00155d1794



pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt



https://blog.csdn.net/aBlueMouse/article/details/78710553
https://blog.csdn.net/u010327061/article/details/80101301

SRCNN:
https://blog.csdn.net/xu_fu_yong/article/details/96434132


EDVR:
https://blog.csdn.net/WinerChopin/article/details/97310843
命令：
C:\Python37\python -m torch.distributed.launch --nproc_per_node=2 --master_port=21688 train.py -opt options/train/train_EDVR_OURS_M.yml --launcher pytorch
C:\Python37\python train.py -opt options/train/train_EDVR_OURS_M.yml