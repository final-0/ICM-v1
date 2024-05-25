<div align="center">

### Scalable Image Coding for Humans and Machines<br>
### Image Coding for Machines<br>
</div>

This is the official pytorch implementation of the following papers.
This is the Pytorch repository of the papers<br>
・Image Coding for Machines with Edge Information Learning Using Segment Anything
([arXiv](https://arxiv.org/abs/2403.04173))<br>
・Scalable Image Coding for Humans and Machines Using Feature Fusion Network
([arXiv](https://arxiv.org/abs/2405.09152))<br>

<div align="center">
  
### Installation
</div>

<div align="center">
  
### Usage
</div>

Download [model checkpoints](https://drive.google.com/drive/folders/1wqK1HXZ4Ua3jqo2GHkxHJzIojsueI3xx?usp=drive_link). You can obtain "ica.pth.tar" and "icm.pth.tar" from this link. These checkpoints can be used by placing them in the "param" folder.<br>
``` 
param ---- param_details.txt
       |-- icm.pth.tar (image compression model for Machines)
       |-- ica.pth.tar (additional information compression model)
```
If you want to compress images for "Machines", run the following command :
``` 
python3 coding_m.py --checkpoint param/icm.pth.tar --input image/input
```
If you want to compress images for "Humans" & "Machines", run the following command :
``` 
python3 coding_hm.py --checkpoint_m param/icm.pth.tar --checkpoint_a param/ica.pth.tar --input image/input
```

