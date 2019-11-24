import torch
import torch.backends.cudnn as cudnn
import os
from PIL import Image
import numpy as np
from SR.SRCNN_Pyorch_1 import SRCNN as SRCNN
from SR.SRCNN_Pyorch_1 import transform_data as transform_data

weights_file = 'SRCNN_weights/epoch_2_.pth'
scale = 4
image_path = 'J:/img_540p'
out_path = 'J:/output'
if __name__ == '__main__':
    cudnn.benchmark = True
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for file in os.listdir(image_path):
        in_file = os.path.join(image_path,file)
        out_file = os.path.join(out_path,file)
        image = Image.open(in_file).convert('RGB')
        image = image.resize((image.width * scale, image.height * scale), resample=Image.BICUBIC)

        input = transform_data(image).to(device)
        input = input.unsqueeze(0)
        with torch.no_grad():
            preds = model(input).clamp(0.0, 1.0)

        preds = preds.mul(255.0).cpu().numpy().squeeze(0)
        output = preds.transpose([1, 2, 0])
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        print(output.shape)
        output = Image.fromarray(output,'RGB')
        output.save(out_file)

        break




