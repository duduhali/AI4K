

./naic_submit -token 2fef0f957ee8ed2b -file ./SRCNN.zip


BasicSR:
https://blog.csdn.net/Arthur_Holmes/article/details/103372633

EDVR:
https://blog.csdn.net/WinerChopin/article/details/96427327


python -m torch.distributed.launch --nproc_per_node=2 --master_port=21688 train.py -opt options/train/train_EDVR_OURS_M.yml --launcher pytorch
