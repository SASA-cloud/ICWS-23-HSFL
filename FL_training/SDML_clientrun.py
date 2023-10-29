import torch
import socket
import time
import multiprocessing
import os
import argparse
import pickle

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Client import Client
import utils.config as config
import utils.utils as utils
from LinUCB import muLinUCB

index = config.index

datalen = config.datalen_each
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Preparing Client')
client = Client(index, config.CLIENTS_LIST[0], datalen, config.model_name)
# ucb_agent 
ucb_agent = muLinUCB(layerInfo=config.layer_info,privacyLeakage=config.privacy_leakage,actionList=config.actionList)


train_algo = config.args.train_algo
offload = False if train_algo=='FL' else True
first = True # First initializaiton control


logger.info('Preparing Data.')
trainloader, classes= utils.get_local_dataloader(index, 0)

logger.info("training algorithm is"+train_algo)

flag = False # Bandwidth control flag.

res = {'C_forward':[],'smashed':[],'C_backward':[],'aggregate':[],'split_layer':[],'C_per_round_infer':[],'C_per_round_agg':[],'C_gradient':[]}

for r in range(config.R): # FL rounds
	logger.info('====================================>')
	logger.info('ROUND: {} START'.format(r))


	logger.info('==> initialization for Round : {:}'.format(r))
	
	
	s_time_rebuild = time.time()

	if config.args.latency_decomposition:
		config.split_layer=[config.layer_wise_latency_sp[r]]
	elif not first and train_algo == "FedAdapt": 
		config.split_layer = client.recv_msg(client.sock)[1]
		
	elif train_algo == 'SFLv1' or train_algo == 'SFLv2':
		config.split_layer = config.SFL_layer
	elif train_algo == 'HSFL':
		sp = ucb_agent.getEstimationAction()
		config.split_layer = [sp] * config.K

		msg = ['SPLIT_LAYERS',sp]
		client.send_msg(client.sock, msg)
	else: # FL
		split_layers = config.split_layer
	res['split_layer'].append(config.split_layer[index])
	
	client.initialize(config.split_layer[index], offload, first, LR,train_algo)

	if first:
		first = False

	e_time_rebuild = time.time()
	logger.info('Initialize time: ' + str(e_time_rebuild - s_time_rebuild))
	logger.info('==> initialization Finish')

	# training
	s_time_infer = time.time()
	infer_time_total,train_time_list = client.train(trainloader)
	e_time_infer = time.time()
	logger.info('ROUND: {} END'.format(r))
	
	# aggregation
	s_time_agg = time.time()
	logger.info('==> Waiting for aggregration')
	upload_time = client.upload()
	e_time_agg = time.time()

	time_list = train_time_list[:]
	time_list.append(upload_time) # [forward,smashed,backward,aggregate]
	
	# res
	res['C_forward'].append(time_list[0])
	res['smashed'].append(time_list[1])
	res['C_backward'].append(time_list[2])

	res['aggregate'].append(time_list[3])

	res['C_per_round_infer'].append(e_time_infer-s_time_infer)
	res['C_per_round_agg'].append(e_time_agg-s_time_agg)


	if train_algo == 'HSFL':
		ucb_agent.updateA_b(config.split_layer[index],infer_time_total)
	logger.info('Round Finish')

	msg = ['MSG_RES_CLIENT_TO_SERVER',config.self_python_key,res]
	client.send_msg(client.sock,msg)

	if r > 49: 
		LR = config.LR * 0.1



	