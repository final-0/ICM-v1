import torch
import torch.nn.functional as F
from torchvision import transforms
from models2 import ICM
from models2 import ICA
import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2
from utils import *

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--checkpoint_m", type=str, help="Path to a checkpoint")
    parser.add_argument("--checkpoint_a", type=str, help="Path to a checkpoint")
    parser.add_argument("--input", type=str, help="Path to input_images")
    args = parser.parse_args(argv)
    return args

def main(argv):
    os.makedirs("image/output_machines",exist_ok=True)
    os.makedirs("image/output_humans",exist_ok=True)
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
    ica = ICA()
    ica = ica.to(device)
    ica.eval()
    
    PSNR = 0
    Bit_rate_m = 0
    Bit_rate_a = 0
    
    dictory = {}
    if args.checkpoint_a:
        print("Loading : additional-information model", args.checkpoint_a)
        checkpoint_a = torch.load(args.checkpoint_a, map_location=device)
        for k, v in checkpoint_a["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        ica.load_state_dict(dictory)
    if args.checkpoint_m:
        print("Loading : machine model", args.checkpoint_m)
        checkpoint_m = torch.load(args.checkpoint_m, map_location=device)
        for k, v in checkpoint_m["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        icm.load_state_dict(dictory)
    
    
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
            output_machines_y_hat = output_machines["y_hat"]
            output_humans = ica.forward(x_padded,output_machines_y_hat)

            output_machines['x_hat'].clamp_(0, 1)
            output_machines["x_hat"] = crop(output_machines["x_hat"], padding)
            print(f'Bit-rate_m: {compute_bpp(output_machines):.3f}bpp')
            Bit_rate_m += compute_bpp(output_machines)

            output_humans['x_hat'].clamp_(0, 1)
            output_humans["x_hat"] = crop(output_humans["x_hat"], padding)
            print(f'Bit-rate_a: {compute_bpp(output_humans):.3f}bpp')
            Bit_rate_a += compute_bpp(output_humans)
            PSNR += compute_psnr(x, output_humans["x_hat"])
            print(f'PSNR: {compute_psnr(x, output_humans["x_hat"]):.2f}dB')

            yi = output_machines["x_hat"].detach().cpu().numpy()
            yi = np.squeeze(yi[0,:,:,:])*255
            yi = yi.transpose(1,2,0)
            yi = cv2.cvtColor(yi, cv2.COLOR_RGB2BGR)
            img_PATH = img_path[-16:-4]
            cv2.imwrite('image/output_machines/%s.png'%(img_PATH),yi.astype(np.uint8))
            yi = output_humans["x_hat"].detach().cpu().numpy()
            yi = np.squeeze(yi[0,:,:,:])*255
            yi = yi.transpose(1,2,0)
            yi = cv2.cvtColor(yi, cv2.COLOR_RGB2BGR)
            img_PATH = img_path[-16:-4]
            cv2.imwrite('image/output_humans/%s.png'%(img_PATH),yi.astype(np.uint8))
        
    print('\n---result_bpp(estimate)---')
    print(f'average_Bit-rate : --------machines-------- : {(Bit_rate_m/len(img_list)):.3f} bpp')
    print(f'average_Bit-rate : -additional information- : {(Bit_rate_a/len(img_list)):.3f} bpp')

    print('--- save image ---')
    print('compressed images are saved in "image/output_machines" and "image/output_humans"\n')
    print(f'average_PSNR: {(PSNR/len(img_list)):.2f}dB')

if __name__ == "__main__":
    print('\n::: compress images for Humans and Machines :::\n')
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    