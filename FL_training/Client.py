import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import sys

sys.path.append('../')
import utils.config as config
import utils.utils as utils
from utils.Communicator import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class Client(Communicator):
	def __init__(self, index, client_key, datalen, model_name):
		super(Client, self).__init__(index, client_key)
		self.datalen = datalen
		self.device = 'cuda:0' if torch.cuda.is_available() and config.args.gpu==True else 'cpu'
		logger.info("self.device="+str(self.device))
		# self.device = 'cpu'
		self.model_name = model_name
		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)

		logger.info('Connecting to Server.')
		# self.sock.connect((server_addr,server_port))
		self.sock = config.client_sock

	def initialize(self, split_layer, offload, first, LR, train_algo):
		self.split_layer = split_layer
		if offload or first:

			logger.debug('Building Model.')
			self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, config.model_cfg)
			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()

		self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)
		logger.debug('Receiving Global Weights..')
		weights = self.recv_msg(self.sock,'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT')[1]
		if config.args.latency_decomposition:
			msg1 = ['ok_msg','ok']
			self.send_msg(self.sock,msg1)
		s_split_model_time = time.time()
		
		print(self.net)
		
		if train_algo == "FedAdapt": 
			if self.split_layer == (config.model_len -1):
				self.net.load_state_dict(weights)
			else:
				pweights = utils.split_weights_client(weights,self.net.state_dict())
				self.net.load_state_dict(pweights)
		else:
			self.net.load_state_dict(weights) # load weights
		# logger.debug('Initialize Finished')

		e_split_model_time = time.time()
		logger.debug('Initialize Finished')
		return e_split_model_time - s_split_model_time

	def train(self, trainloader):
		if config.args.test_network:
			# Network speed test
			network_time_start = time.time()
			msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
			self.send_msg(self.sock, msg)
			msg = self.recv_msg(self.sock,'MSG_TEST_NETWORK')[1]
			network_time_end = time.time()
			network_speed = (2 * config.model_size * 8) / (network_time_end - network_time_start) #Mbit/s 

			logger.info('Network speed is {:}'.format(network_speed))
			msg = ['MSG_TEST_NETWORK', self.key, network_speed]
			self.send_msg(self.sock, msg)

		# Training start
		s_time_total = time.time()
		self.net.to(self.device)
		self.net.train()
		
		# time statistics 
		s_time_forward,e_time_forward = 0.0,0.0  # forward
		s_time_backward,e_time_backward = 0.0,0.0 # backpropagation
		s_time_tx_smashed,e_time_tx_smashed = 0.0,0.0 # smashed data transmission
		print("self.split_layer= ", self.split_layer)
		if self.split_layer == (config.model_len -1): # No offloading training
			print('non offloading')
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()

				s_time_forward = time.time()
				outputs = self.net(inputs)
				e_time_forward = time.time()

				loss = self.criterion(outputs, targets)

				s_time_backward = time.time()
				loss.backward()
				self.optimizer.step()
				e_time_backward = time.time()
			
		else: # Offloading training
			print('offloading')
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()

				s_time_forward = time.time()
				outputs = self.net(inputs)
				e_time_forward = time.time()

				msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu().detach().clone(), targets.cpu()]
				s_time_tx_smashed = time.time()
				self.send_msg(self.sock, msg)
				if config.args.latency_decomposition:
					self.recv_msg(self.sock)
				e_time_tx_smashed = time.time()

				# Wait receiving server gradients
				msg = self.recv_msg(self.sock)
				gradients = msg[1].to(self.device)
				if config.args.latency_decomposition:
					msg1 = ['ok_msg','ok']
					self.send_msg(self.sock,msg1)

				# print("gradients=: ", gradients)
				# gradients = gradients.to(self.device)
				s_time_backward = time.time()
				outputs.backward(gradients)
				self.optimizer.step()
				e_time_backward = time.time()

		e_time_total = time.time()
		logger.info('Total time: ' + str(e_time_total - s_time_total))
		# Nbatch
		training_time_pr = (e_time_total - s_time_total) / config.NBatch 
		logger.info('training_time_per_iteration: ' + str(training_time_pr))

		msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.key, training_time_pr]
		self.send_msg(self.sock, msg)

		# time_dict = {'total': e_time_total - s_time_total,
		# 			'forward': e_time_forward - s_time_forward,
		# 			'smashed': e_time_tx_smashed - s_time_tx_smashed,
		# 			'backward': e_time_backward - s_time_backward}
		time_list = [
			(e_time_forward - s_time_forward)*config.NBatch,
			(e_time_tx_smashed - s_time_tx_smashed)*config.NBatch,
			(e_time_backward - s_time_backward)*config.NBatch,
			]
		return training_time_pr,time_list
	
	def upload(self):

		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
		s_time_tx_client_model = time.time()
		self.send_msg(self.sock, msg)
		if config.args.latency_decomposition:
			self.recv_msg(self.sock)
		e_time_tx_client_model = time.time()

		return e_time_tx_client_model - s_time_tx_client_model

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

	def select_sp(self):
		pass

	def update_sp(self):
		pass

