# Meta-Weight-Net
NeurIPS'19: Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting (Official Pytorch implementation for noisy labels).
The implementation of class imbalance is available at https://github.com/xjtushujun/Meta-weight-net_class-imbalance.


================================================================================================================================================================


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
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 0.4.0 
- Torchvision 0.2.0


## Running Meta-Weight-Net on benchmark datasets (CIFAR-10 and CIFAR-100).
Here is an example:
```bash
python train_WRN-28-10_Meta_PGC.py --dataset cifar10 --corruption_type unif(flip2) --corruption_prob 0.6
```

The default network structure is WRN-28-10, if you want to train with ResNet32 model, please reset the learning rate delay policy.

A stable version is relased.
```bash
python MW-Net.py --dataset cifar10 --corruption_type unif(flip2) --corruption_prob 0.6
```



## Acknowledgements
We thank the Pytorch implementation on glc(https://github.com/mmazeika/glc) and learning-to-reweight-examples(https://github.com/danieltan07/learning-to-reweight-examples).


Contact: Jun Shu (xjtushujun@gmail.com); Deyu Meng(dymeng@mail.xjtu.edu.cn).




