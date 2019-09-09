# meta-weight-net
NeurIPS'19: Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting (Pytorch implementation).


==========================================================================================================


This is the code for the paper:
[Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://arxiv.org/abs/1902.07379)  
Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, Deyu Meng*
To be presented at [NeurIPS 2019](https://nips.cc/Conferences/2019/).  

If you find this code useful in your research then please cite  
```bash
@inproceedings{han2018coteaching,
  title={Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting},
  author={Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu},
  booktitle={NeurIPS},
  year={2019}
}
``` 


## Setups
All code was developed and tested on a single machine equiped with a NVIDIA K80 GPU. The environment is as bellow:  

- Linux 
- Python 3
- PyTorch 0.4.0 
- Torchvision 0.2.0

```bash
python train_WRN-28-10_Meta_PGC.py --dataset cifar10 --corruption_type unif --corruption_prob 0.6
```

The default network structure is WRN-28-10, if you want to train with ResNet32 model, please reset the learning rate delay policy.
