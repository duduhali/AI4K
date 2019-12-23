

./naic_submit -token 2fef0f957ee8ed2b -file ./SRCNN.zip


BasicSR:
https://blog.csdn.net/Arthur_Holmes/article/details/103372633

EDVR:
https://blog.csdn.net/WinerChopin/article/details/96427327
cd /models/archs/dcn    sudo python3 setup.py develop


python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=21688 train.py -opt options/train/train_EDVR_OURS_M.yml --launcher pytorch



DataParallel(
  (module): EDVR(
    (conv_first): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (feature_extraction): Sequential(
      (0): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (4): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (fea_L2_conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (fea_L2_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fea_L3_conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (fea_L3_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (pcd_align): PCD_Align(
      (L3_offset_conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L3_offset_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L3_dcnpack): ModulatedDeformConvPack(
        (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (L2_offset_conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L2_offset_conv2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L2_offset_conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L2_dcnpack): ModulatedDeformConvPack(
        (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (L2_fea_conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L1_offset_conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L1_offset_conv2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L1_offset_conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (L1_dcnpack): ModulatedDeformConvPack(
        (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (L1_fea_conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cas_offset_conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cas_offset_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cas_dcnpack): ModulatedDeformConvPack(
        (conv_offset_mask): Conv2d(64, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (tsa_fusion): TSA_Fusion(
      (tAtt_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (tAtt_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fea_fusion): Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
      (sAtt_1): Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (avgpool): AvgPool2d(kernel_size=3, stride=2, padding=1)
      (sAtt_2): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
      (sAtt_3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (sAtt_4): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (sAtt_5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (sAtt_L1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (sAtt_L2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (sAtt_L3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (sAtt_add_1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (sAtt_add_2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (recon_trunk): Sequential(
      (0): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (4): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (5): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (6): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (7): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (8): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (9): ResidualBlock_noBN(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (upconv1): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (upconv2): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (pixel_shuffle): PixelShuffle(upscale_factor=2)
    (HRconv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv_last): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
  )
)
