import argparse
from model.RCAN3D import RCAN3D



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()


    args.n_frames = 8
    args.n_res_blocks = 4
    args.n_resgroups = 2
    args.n_feats = 64
    args.reduction = 16
    args.n_colors = 3
    args.scale = 4
    args.rgb_range = 255



    model = RCAN3D(args)
    print(model)






