import sys
sys.path.append('../')

import pickle
import os
from utils.Communicator import recv_index_from_server,get_clients_info,send_index_to_clients

import socket
import argparse

parser = argparse.ArgumentParser()

# env
parser.add_argument('--env', type=str, default="edge")
parser.add_argument('--train_algo', help='training algorithms', type=str, default='FL') # HSFL, FedAdapt, FL, SplitFedv1, SplitFedv2
parser.add_argument('--SFL_sp',type = int, default=5) # split point for SplitFedv1 and SplitFedv2
parser.add_argument('--model_name',type = str, default='VGG9') # model name, VGG5 or VGG9
parser.add_argument('--gpu',type=bool,default=False) # use GPU accelerator?
parser.add_argument('--test_network',type=bool,default=True) # test network speed
parser.add_argument('--latency_decomposition',type=bool,default=False) # mode of measuring the proportion of components of latency
parser.add_argument('--pre_train',type=bool,default=False) # using pretrained target model?
parser.add_argument('--privacy',type=bool,default=False) # inverse efficacy?
args = parser.parse_args()
print("args.gpu: ",args.gpu)

# FL training configration
R = 2 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size

# layer_wise_latency_sp=[1,4,7,9,10,13] # vgg9
# layer_wise_latency_sp=[1,3,4,5,6] # vgg5
layer_wise_latency_sp=[args.SFL_sp] 

if args.env == 'hpc':
	K = 25 # Number of devices 
	G = 5
elif args.env == 'edge':
	K = 5
	G = 3
elif args.env == 'sys':
	K = 1 # Number of devices 
	G = 1 # Number of groups

N = 50000 # data length
# datalen for each client:
# NBatch = int((N / (K * B)))
# datalen_each = int(N / K)

# single client
NBatch = int((N / (K * B)/5))
datalen_each = int(10000)

# Network configration
self_python_hostname=socket.gethostname()
self_python_ip = socket.gethostbyname(self_python_hostname)

# server IP
if args.env=='hpc':
	SERVER_ADDR = '10.10.16.21' 
elif args.env=='edge':
	SERVER_ADDR = '192.168.3.5' # laptop
	if self_python_ip=='127.0.0.1':
		self_python_ip=SERVER_ADDR 
elif args.env=='sys':
	# SERVER_ADDR = '10.0.0.27' # 83
	SERVER_ADDR = '10.0.0.23' # 25 
	if self_python_ip.split('.')[0]=='127':
		self_python_ip = socket.gethostbyname("FedAdapt-drj")

pid = os.getpid()
self_python_key = self_python_ip+":"+ str(pid)  # ip:pid
print('self_python_key: ',self_python_key)

SERVER_PORT = 50000
lserver_addr = (SERVER_ADDR,SERVER_PORT)


CLIENTS_LIST,client_sockets_dict,server_sock = [],{},None
iteration,index,client_sock={},None,None

if self_python_ip == SERVER_ADDR: # for server
	CLIENTS_LIST, client_sockets_dict, server_sock = get_clients_info(K,lserver_addr)
	send_index_to_clients(client_sockets_dict)
	iteration = {client_key:5 for client_key in CLIENTS_LIST}  # local training times for each device
else: # client
	client_sock,client_key, index, itr = recv_index_from_server(lserver_addr)
	iteration[client_key]= itr
	CLIENTS_LIST.append(client_key)

# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0),  # 1
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), #3
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), #4
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), #5
	('D', 128, 10, 1, 10, 128*10)],#6
	'VGG9':[#[1,4,7,9,10,11,12,13]
		('C',3,64,3,65535,1769472), 
		('M',64,64,2,16384,0), # 1
		('C',64,64,3,16384,9437184), 
		('C',64,128,3,32768,18874368), 
		('M',128,128,2,8192,0),# 4
		('C',128,128,3,8192,9437184),
		('C',128,256,3,16384,18874368),
		('M',256,256,2,4096,0), #7
		('C',256,256,3,4096,9437184),
		('M',256,256,2,1024,0), #9
		('C',256,256,3,1024,2359296), #10
		('D',1024,4096,1,4096,4194304), # 11
		('D',4096,4096,1,4096,16777216),  # 12
		('D',4096,10,1,10,40960), # 13
	]
}

privacy = True

model_name = args.model_name
if model_name == 'VGG5':
	actionList =  [1,3,4,5,6] # vgg5
	# actionList =  [3,4,5,6] # vgg5
	model_size = 2.3 
	total_flops = 8488192
	split_layer = [6] * K #Initial split layers 
	model_len = 7

	# privacy_leakage = [0.4516166306393763,0.3399455727858786,0.05740828017257315,0.025361912312856838]
	privacy_leakage = [0.552432027525534,0.4516166306393763,0.3399455727858786,0.05740828017257315,0.025361912312856838]
	layer_info = { # action No. : [Flops, smashed_datasize,/ client_side_model_size]
		# 0:[884736,32768],
		0:[884736,8192], # 1
		# 2:[5603328,16384],
		1:[5603328,4096], # 3
		2:[7962624,4096], # 4
		3:[8486912,128], # 5
		4:[8488192,10], # 6
	}

elif model_name == 'VGG9':
	# actionList =  [1,4,7,9,10,11,12,13] # vgg9
	actionList =  [1,4,7,9,10,13] # vgg9
	model_size = 91.1
	total_flops = 91201536
	split_layer = [13] * K #Initial split layers 
	model_len = 14

	# privacy_leakage = [0.552432027525534,0.4516166306393763,0.3399455727858786,0.05740828017257315,0.025361912312856838,0.025361912312856838,0.025361912312856838,0.025361912312856838] # vgg9
	privacy_leakage = [0.578284082, 0.516490632, 0.369720727, 0.222147408, 0.149785127, 0.033617634] 
	layer_info ={
		0:[1769472,16384], # 1
		1:[30081024,8192], # 4
		2:[58392576,4096], # 7
		3:[67829760,1024], # 9
		4:[70189056,1024], # 10
		5:[91201536,10], # 13
	}

# for SFLv1 and SFLv2
SFL_layer = [args.SFL_sp] * K  

# Dataset configration 
dataset_path = '../../data/cifar/'
print("data_path: ",dataset_path)

# results saving path
trail_num = '' # results saving directory
ppo_model_path =  '../results/'+trail_num+'PPO.pth'
labels_path = '../results/'+trail_num+'labels.pkl'
vgg_model_path = '../results/'+trail_num+model_name+'.pth'
RL_res_path = '../results/'+trail_num+'RL_res.pkl'
FL_res_path = f'../results/'+trail_num+args.train_algo+'_res.pkl' 
client_list_path = '../results/'+trail_num+'client_list.pkl'



# RL training configration
max_episodes = 500      # max training episodes
max_timesteps = 100       # max timesteps in one episode
exploration_times = 20	   # exploration times without std decay
n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timesteps 
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_b = 100				   # Batchsize
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)

random = True
random_seed = 0
