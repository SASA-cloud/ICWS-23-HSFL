
### 1. Introduction
This repository contains code of "HSFL: Efficient and Privacy-Preserving Offloading for Split and Federated Learning in IoT Services" on ICWS 2023.

This repository implements the following frameworks: HSFL (do not include hierarhchial HSFL), FedAdapt, SplitFedv1, SplitFedv2, and FL on truely (physically) distributed platform, where nodes are communicate with each other using python sockets.


### 2. Structure of this code:
```
"data": CIFAR10 dataset
"RL_training": training FedAdapt RL agent.
"FL_training": code for running HSFL, FedAdapt, SplitFedv1, SplitFedv2, and FL.
"models": the target training model (VGG) and models for FedAdapt RL agent.
"results": running results.
"HSFL_experiments": data analysis for the paper.
"utils": utilities.
"environment": building environment.
```

### 3. Environment
* Edge server:
a mac laptop using anaconda3 to build a virtual environment.
you can create this environment by executing:
```bash
conda env create -n pytorch -f pytorch.yaml
```
* End devices:
we use three Raspberry Pi 4B and two Jetson Nano.
Virtualenv are used instead of anaconda3 to build a virtual environment.
We build up torch 1.8.0 and torchvision 0.9.0 environemnts with wheels.
The environment building process is very tiring, so we leave the shell scrips we used in the `envronment/torch_on_clients.sh` and some references here:
* PyTorch for Jetson - Jetson & Embedded Systems / Jetson Nano - NVIDIA Developer Forums: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
* Jetson Nano: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048
* Pytorch (Pi): https://discuss.pytorch.org/t/installing-pytorch-on-raspberry-pi-3/25215/12
* Pyotrch: https://github.com/FedML-AI/FedML-IoT/tree/master/pytorch-pkg-on-rpi
* Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048
* README of `SAMIKWA E, MAIO A D, BRAUN T. Ares: Adaptive resource-aware split learning for internet of things. Computer Networks, 2022, 218.`
* README of `Gao, Y., et al. (2020). End-to-End Evaluation of Federated Learning and Split Learning for Internet of Things. 2020 International Symposium on Reliable Distributed Systems (SRDS): 91-100.`
* README of `C. Thapa, M. A. P. Chamikara, and S. A. Camtepe, "Advancements of federated learning towards privacy preservation: from federated learning to split learning," in Federated Learning Systems: Springer, 2021, pp. 79-109.`

* Code modification:
We provide the configration and source code of those frameworks in our paper. If you want to reproduce the results based on your own devices, please modifify those IPs, file routes, sockets numbers in the `utils/config.py`.

### 4. Run the code:
you should download the code on each nodes, and then:

* the following commands help you to conduct HSFL, FedAdapt, SplitFedv1, SplitFedv2, and FL on physically connected single-layer devices.

Be sure to execute the command on the server first and then the clients.

for FedAdapt, you should first train a RL agent:
```bash
cd RL_training
python RL_serverrun.py --env edge --gpu [use_gpu] # on server
python RL_clientrun.py --env edge --gpu [use_gpu] # on client
```
the `[use_gpu]` can be True or False, indicating where the the gpu accelerators are used.

then for all frameworks, you can execute:
```bash
cd FL_training
python SDML_clientrun.py --train_algo [frameworks]  --env edge --gpu [use_gpu] # on server
python SDML_clientrun.py --train_algo [frameworks]  --env edge --gpu [use_gpu] # on client
```
`[use_gpu]` is the the same to above.

`[frameworks]` can be FL, SFLv2, HSFL, SFLv1, FedAdapt, indicating FL, SplitFedv2, HSFL, SplitFedv1, FedAdapt, and FL respectively.


### 5. Citation

Please cite the paper as follows: 

```
# IEEE
R. Deng, X. Du, Z. Lu, Q. Duan, S.-C. Huang, and J. Wu, "HSFL: Efficient and Privacy-Preserving Offloading for Split and Federated Learning in IoT Services," presented at the 2023 IEEE International Conference on Web Services (ICWS), 2023.

# Bibtex
@inproceedings{deng2023hsfl,
  title={HSFL: Efficient and Privacy-Preserving Offloading for Split and Federated Learning in IoT Services},
  author={Deng, Ruijun and Du, Xin and Lu, Zhihui and Duan, Qiang and Huang, Shih-Chia and Wu, Jie},
  booktitle={2023 IEEE International Conference on Web Services (ICWS)},
  pages={658--668},
  year={2023},
  organization={IEEE}
}
```

### 6. Acknowledgements
HSFL partially used code from the following papers:
* D. Wu, R. Ullah, P. Harvey, P. Kilpatrick, I. Spence, and B. Varghese, "Fedadapt: Adaptive offloading for iot devices in federated learning," IEEE Internet of Things Journal, 2021, doi: 10.1109/JIOT.2022.3176469.
* C. Thapa, M. A. P. Chamikara, S. Camtepe, and L. Sun, "Splitfed: When federated learning meets split learning," arXiv preprint arXiv:2004.12088, 2020.
* L. Zhang, L. Chen, and J. Xu, "Autodidactic Neurosurgeon: Collaborative Deep Inference for Mobile Edge Intelligence via Online Learning," presented at the Proceedings of the Web Conference 2021, 2021.
* Y. Venkatesha, Y. Kim, L. Tassiulas, and P. Panda, "Federated Learning With Spiking Neural Networks," IEEE Transactions on Signal Processing, vol. 69, pp. 6183-6194, 2021, doi: 10.1109/tsp.2021.3121632.

