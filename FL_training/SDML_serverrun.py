
import time
import torch
import pickle
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Sever import Sever
import utils.config as config
import utils.utils as utils
import models.PPO as PPO

LR = config.LR
train_algo = config.args.train_algo
offload = False if train_algo=='FL' else True
first = True # First initializaiton control

logger.info('Preparing Sever.')
sever = Sever(0, config.SERVER_ADDR, config.model_name)


state_dim = 2*config.G
action_dim = config.G

if train_algo == 'FedAdapt': 
	# Initialize trained RL agent 
	agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)
	agent.policy.load_state_dict(torch.load(config.ppo_model_path))

logger.info("training algorithm is"+train_algo)

res = {}
res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
res['S_forward'], res['gradients'], res['S_backward'], res['initial'] = [],[],[],[]
res['clients_res'] = {}

# FL rounds
for r in range(config.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))

	logger.info('==> initialization for Round : {:}'.format(r))

	if config.args.latency_decomposition:
		config.split_layer=[config.layer_wise_latency_sp[r]]
	elif not first and train_algo=='FedAdapt': 
		config.split_layer = sever.adaptive_offload(agent, state)
		
	elif train_algo == 'SFLv1' or train_algo == 'SFLv2':
		config.split_layer = config.SFL_layer
	elif train_algo == 'HSFL':
		split_layer = []
		for s in sever.client_socks:
			msg = sever.recv_msg(sever.client_socks[s], 'SPLIT_LAYERS')
			split_layer.append(msg[1])
		config.split_layer = split_layer
	else: 
		split_layers = config.split_layer
	logger.info('split_layers list : '+ str(config.split_layer))

	send_model_time = sever.initialize(split_layers = config.split_layer, LR= LR, train_algo=train_algo)

	if first :
		first = False

	logger.info('==> initialization Finish')

	s_time = time.time() 
	state, bandwidth, time_list = sever.train(train_algo=train_algo, client_ips=config.CLIENTS_LIST) 
	aggregrated_model = sever.aggregate(config.CLIENTS_LIST,train_algo) 
	e_time = time.time() 

	# res
	training_time = e_time - s_time 
	res['training_time'].append(training_time)
	res['bandwidth_record'].append(bandwidth) 

	test_acc = sever.test(r)
	res['test_acc_record'].append(test_acc)

	res['S_forward'].append(time_list[0])
	res['gradients'].append(time_list[1])
	res['S_backward'].append(time_list[2])
	res['initial'].append(send_model_time)


	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(training_time))
	
	client_res_dict = {}
	for s in sever.client_socks:
		msg = sever.recv_msg(sever.client_socks[s], 'MSG_RES_CLIENT_TO_SERVER')
		client_res_dict[msg[1]]=msg[2]
	res['clients_res'] = client_res_dict


	with open(config.FL_res_path,'wb') as f: 
		pickle.dump(res,f)


	if r > 49:
		LR = config.LR * 0.1

