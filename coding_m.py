import torch
import torch.nn.functional as F
from torchvision import transforms
from models import ICM
import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2
from utils import *

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--input", type=str, help="Path to input_images")
    parser.add_argument("--real", action="store_true", default=False)
    args = parser.parse_args(argv)
    return args

def main(argv):
    os.makedirs("image/output_machines",exist_ok=True)
    args = parse_args(argv)
    p = 128
    path = args.input
    img_list = []
    for file in os.listdir(path):
        img_list.append(file)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    icm = ICM()
    icm = icm.to(device)
    icm.eval()
    Bit_rate = 0
    
    dictory = {}
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        icm.load_state_dict(dictory)

    if args.real:
        print('\n::real compression::\n')
        icm.update()
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            print(img_path[-16:])
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                out_enc = icm.compress(x_padded)
                out_dec = icm.decompress(out_enc["strings"], out_enc["shape"])
                
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
                
                Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                yi = out_dec["x_hat"].detach().cpu().numpy()
                yi = np.squeeze(yi[0,:,:,:])*255
                yi = yi.transpose(1,2,0)
                yi = cv2.cvtColor(yi, cv2.COLOR_RGB2BGR)
                img_PATH = img_path[-16:-4]
                cv2.imwrite('image/output_machines/%s.png'%(img_PATH),yi.astype(np.uint8))

        print('\n---result_bpp(real)---')
        print(f'average_Bit-rate: {(Bit_rate/len(img_list)):.3f} bpp')
        print('--- save image ---')
        print('compressed images are saved in "image/output_machines"\n')

    else:
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            print(img_path[-16:])
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)

            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                output_machines = icm.forward(x_padded)
                
                output_machines['x_hat'].clamp_(0, 1)
                output_machines["x_hat"] = crop(output_machines["x_hat"], padding)
                print(f'Bit-rate: {compute_bpp(output_machines):.3f}bpp')

                Bit_rate += compute_bpp(output_machines)
                yi = output_machines["x_hat"].detach().cpu().numpy()
                yi = np.squeeze(yi[0,:,:,:])*255
                yi = yi.transpose(1,2,0)
                yi = cv2.cvtColor(yi, cv2.COLOR_RGB2BGR)
                img_PATH = img_path[-16:-4]
                cv2.imwrite('image/output_machines/%s.png'%(img_PATH),yi.astype(np.uint8))

        print('\n---result_bpp(estimate)---')
        print(f'average_Bit-rate: {(Bit_rate/len(img_list)):.3f} bpp')
        print('--- save image ---')
        print('compressed images are saved in "image/output_machines"\n')

if __name__ == "__main__":
    print('\n::: compress images for Machines :::\n')
    print(torch.cuda.is_available())
    main(sys.argv[1:])