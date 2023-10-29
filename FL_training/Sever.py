

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import random
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import time
import sys
sys.path.append('../')
from utils.Communicator import *
import utils.utils as utils
import utils.config as config

np.random.seed(0)
torch.manual_seed(0)

class Sever(Communicator):
	def __init__(self, index, ip_address, model_name):
		super(Sever, self).__init__(index, ip_address)
		self.device = 'cuda:0' if torch.cuda.is_available() and config.args.gpu==True else 'cpu'
		logger.info("self.device="+str(self.device))
		# self.device = 'cpu'
		# self.port = server_port
		self.model_name = model_name

		# hpc sock
		self.sock = config.server_sock
		self.client_socks = config.client_sockets_dict

		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)
		if config.args.pre_train:
			print('load pre trained model')
			self.uninet.load_state_dict(torch.load(config.vgg_model_path))
			self.uninet = self.uninet.to(self.device)
		

		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])		
		logger.info("get testing dataset...")
		self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=True, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=0)
 		
	def initialize(self, split_layers,  LR, train_algo):  
		self.split_layers = split_layers
		config.split_layer = split_layers
		self.criterion = nn.CrossEntropyLoss()

		##################constructing client-side model####################
		s_sp_model_time = time.time()
		e_sp_model_time = time.time()

		if train_algo == 'FedAdapt' or  train_algo == 'FL' :
			self.nets = {}
			self.optimizers= {}


			for i in range(len(split_layers)):
				client_ip = config.CLIENTS_LIST[i]
				self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)
				if split_layers[i] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					#offloading weight in server also need to be initialized from the same global weight
					cweights = utils.get_model('Client', self.model_name, split_layers[i], self.device, config.model_cfg).state_dict()
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)

					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR, momentum=0.9)
				self.nets[client_ip]=self.nets[client_ip].to(self.device)


				msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.cpu().state_dict()]
				s_time_tx_client_model = time.time()
				self.send_msg(self.client_socks[client_ip], msg)
				e_time_tx_client_model = time.time()
				# print('ok')



		elif train_algo == 'SFLv1': 
			self.nets = {}
			self.optimizers= {}


			for i in range(len(split_layers)):
				client_ip = config.CLIENTS_LIST[i]
				self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)
				if split_layers[i] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					#offloading weight in server also need to be initialized from the same global weight
					cweights = utils.get_model('Client', self.model_name, split_layers[i], self.device, config.model_cfg).state_dict()
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)
					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR, momentum=0.9)
				self.nets[client_ip]=self.nets[client_ip].to(self.device)

		elif  train_algo == 'SFLv2':  
			self.net = utils.get_model('Server',self.model_name, split_layers[0], self.device, config.model_cfg)
			if split_layers[0] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
				cweights = utils.get_model('Client', self.model_name, split_layers[0], self.device, config.model_cfg).state_dict()
				pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.net.state_dict())
				self.net.load_state_dict(pweights)
				print(self.net)
				self.optimizer = optim.SGD(self.net.parameters(),lr=LR, momentum=0.9)
			self.net = self.net.to(self.device)
		
		if train_algo == 'SFLv1' or train_algo == 'SFLv2':
			cweights = utils.get_model('Client', self.model_name, split_layers[0], self.device, config.model_cfg).state_dict()
			cweights_send = utils.split_weights_client(self.uninet.cpu().state_dict(),cweights)
			msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', cweights_send]
			for i in self.client_socks:
				s_time_tx_client_model = time.time()
				self.send_msg(self.client_socks[i], msg)
				if config.args.latency_decomposition:
					self.recv_msg(self.client_socks[i])
				e_time_tx_client_model = time.time()
		#########################seding model######################
			
		if train_algo == 'HSFL':  
			self.nets = {}
			self.optimizers= {}

			for i in range(len(split_layers)):
				client_ip = config.CLIENTS_LIST[i]
				self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)
				if split_layers[i] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					#offloading weight in server also need to be initialized from the same global weight
					cweights = utils.get_model('Client', self.model_name, split_layers[i], self.device, config.model_cfg).state_dict()
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)

					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR, momentum=0.9)
					cweights_send = utils.split_weights_client(self.uninet.cpu().state_dict(),cweights)
				else: # non offloading 
					cweights_send = self.uninet.cpu().state_dict()
				self.nets[client_ip]=self.nets[client_ip].to(self.device)

				msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', cweights_send]
				s_time_tx_client_model = time.time()
				self.send_msg(self.client_socks[client_ip], msg)
				e_time_tx_client_model = time.time()

		return e_time_tx_client_model - s_time_tx_client_model

	def train(self, train_algo,  client_ips): 
		time_list = [0.0,0.0,0.0]
		self.bandwidth = {}
		if config.args.test_network: 
			# Network test
			self.net_threads = {}
			for i in range(len(client_ips)):
				# print("network testing: ", client_ips[i])
				self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ips[i],))
				self.net_threads[client_ips[i]].start()

			for i in range(len(client_ips)):
				self.net_threads[client_ips[i]].join()

			self.bandwidth = {}
			for s in self.client_socks:
				msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK')
				self.bandwidth[msg[1]] = msg[2]
				# print("network testing over: ", client_ips[i])

		# Training start 
		if train_algo == "SFLv2":  # SFLv2 
			for i in range(len(client_ips)):
				if self.split_layers[i] == (config.model_len-1): 
					logger.info(str(client_ips[i]) + ' no offloading training start')
					self._thread_training_no_offloading(client_ips[i],train_algo)
					
				else:
					logger.info(str(client_ips[i]) + ' offloading training start')
					time_list = self._thread_training_offloading(client_ips[i], train_algo)
		else:	# SFLv1	
			self.threads = {}
			for i in range(len(client_ips)):
				if self.split_layers[i] == (config.model_len -1):
					self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ips[i],train_algo))
					logger.info(str(client_ips[i]) + ' no offloading training start')
					self.threads[client_ips[i]].start()
				else:
					logger.info(str(client_ips[i]))
					self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading, args=(client_ips[i],train_algo))
					logger.info(str(client_ips[i]) + ' offloading training start')
					self.threads[client_ips[i]].start()

			for i in range(len(client_ips)):
				self.threads[client_ips[i]].join()
				logger.info("training over: "+str(client_ips[i]))

		self.ttpi = {} # Training time per iteration
		for s in self.client_socks:  
			msg = self.recv_msg(self.client_socks[s], 'MSG_TRAINING_TIME_PER_ITERATION')
			self.ttpi[msg[1]] = msg[2]


		state = None
		if train_algo == 'FedAdapt':
			self.group_labels = self.clustering(self.ttpi, self.bandwidth) # 
			# _a, _b, self.group_labels = self.group(self.ttpi, self.bandwidth)
			self.offloading = self.get_offloading(self.split_layers)
			state = self.concat_norm(self.ttpi, self.offloading) 

		return state, self.bandwidth, time_list

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg)

	def _thread_training_no_offloading(self, client_ip,train_algo):
		pass

	def _thread_training_offloading(self, client_ip, train_algo):
		iteration = config.NBatch
		logger.info("iteration num = :"+ str(iteration))
		for i in range(iteration):
			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
			if config.args.latency_decomposition:
				msg1 = ['ok_msg','ok']
				self.send_msg(self.client_socks[client_ip],msg1)

			smashed_layers = msg[1]
			labels = msg[2]
			inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			# inputs.retain_grad() # 

			inputs.requires_grad_()
			if train_algo == 'SFLv2':
				self.optimizer.zero_grad()

				s_time_forward = time.time()
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				e_time_forward = time.time()
				
				s_time_backward = time.time()
				loss.backward()
				self.optimizer.step()
				e_time_backward = time.time()
				

			else: # FedAdapt, SFLv1, HSFL
				self.optimizers[client_ip].zero_grad()

				s_time_forward = time.time()
				outputs = self.nets[client_ip](inputs)
				e_time_forward = time.time()

				loss = self.criterion(outputs, targets)

				s_time_backward = time.time()
				loss.backward()
				self.optimizers[client_ip].step()
				e_time_backward = time.time()


			# Send gradients to client
			msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad.cpu().detach()]
			s_time_tx_gradients = time.time()
			self.send_msg(self.client_socks[client_ip], msg)
			if config.args.latency_decomposition:
				self.recv_msg(self.client_socks[client_ip])
			e_time_tx_gradients = time.time()

		time_list = [
			(e_time_forward-s_time_forward)*config.NBatch,
			(e_time_tx_gradients-s_time_tx_gradients)*config.NBatch, 
			(e_time_backward-s_time_backward)*config.NBatch,
			]

		logger.info(str(client_ip) + ' offloading training end')
		return time_list

	def aggregate(self, client_ips, train_algo):
		w_local_list =[]

		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
			if config.args.latency_decomposition:
				msg1 = ['ok_msg','ok']
				self.send_msg(self.client_socks[client_ips[i]],msg1)
			if config.split_layer[i] != (config.model_len -1): # offloading cat 
				server_net_weights = self.net.cpu().state_dict() if train_algo =='SFLv2' else self.nets[client_ips[i]].cpu().state_dict()
				w_local = (utils.concat_weights(self.uninet.cpu().state_dict(),msg[1],server_net_weights),config.datalen_each)
				w_local_list.append(w_local)
			else:  # fl
				w_local = (msg[1],config.datalen_each)
				w_local_list.append(w_local)
		
		zero_model = utils.zero_init(self.uninet.cpu()).state_dict()
		# print("zero_model.device: ",utils.zero_init(self.uninet).device)
		aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
		
		self.uninet.load_state_dict(aggregrated_model)
		return aggregrated_model


	def test(self, r):
		self.uninet.eval()
		net = self.uninet.to(self.device)
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = net(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		logger.info('Test Accuracy: {}'.format(acc))

		# Save checkpoint.
		torch.save(self.uninet.state_dict(), config.vgg_model_path) 

		return acc

	def clustering(self, state, bandwidth):
		#sort bandwidth in config.CLIENTS_LIST order
		bandwidth_order =[]
		for c in config.CLIENTS_LIST:
			bandwidth_order.append(bandwidth[c])

		with open(config.labels_path,'rb') as f:
			labels = pickle.load(f)

		return labels

	def group(self, baseline, network): 
		from sklearn.cluster import KMeans
		X = []
		index = 0
		netgroup = []
		for c in self.client_socks:
			X.append([baseline[c]])

		# Clustering without network limitation
		kmeans = KMeans(n_clusters=config.G, random_state=0).fit(X)
		cluster_centers = kmeans.cluster_centers_
		labels = kmeans.predict(X) 
		logging.info("infer_state: (baseline) in first" + str(X))
		logging.info("group: "+ str(labels))
		logging.info("client_socks: " + " ".join(self.client_socks))


		'''
		# Clustering with network limitation
		kmeans = KMeans(n_clusters=config.G - 1, random_state=0).fit(X)
		cluster_centers = kmeans.cluster_centers_
		labels = kmeans.predict(X)

		# We manually set Pi3_2 as seperated group for limited bandwidth
		labels[-1] = 2
		'''
		return kmeans, cluster_centers, labels

	def adaptive_offload(self, agent, state):
		action = agent.exploit(state)
		action = self.expand_actions(action, config.CLIENTS_LIST)

		config.split_layer = self.action_to_layer(action)
		logger.info('Next Round OPs: ' + str(config.split_layer))

		msg = ['SPLIT_LAYERS',config.split_layer]
		self.scatter(msg)
		return config.split_layer

	def expand_actions(self, actions, clients_list): # Expanding group actions to each device
		full_actions = []

		for i in range(len(clients_list)):
			full_actions.append(actions[self.group_labels[i]])

		return full_actions

	def action_to_layer(self, action): # Expanding group actions to each device
		#first caculate cumulated flops
		model_state_flops = []
		cumulated_flops = 0

		for l in config.model_cfg[config.model_name]:
			cumulated_flops += l[5]
			model_state_flops.append(cumulated_flops)

		model_flops_list = np.array(model_state_flops)
		model_flops_list = model_flops_list / cumulated_flops

		split_layer = []
		for v in action:
			idx = np.where(np.abs(model_flops_list - v) == np.abs(model_flops_list - v).min()) 
			idx = idx[0][-1]
			if config.model_name=='VGG5' and idx >= 5: # all FC layers combine to one option
				idx = 6
			elif config.model_name == 'VGG9' and idx >= 11:
				idx = 13
			split_layer.append(idx)
		return split_layer

	def concat_norm(self, ttpi, offloading): 
		ttpi_order = []
		offloading_order =[]
		for c in config.CLIENTS_LIST:
			ttpi_order.append(ttpi[c])
			offloading_order.append(offloading[c])

		group_max_index = [0 for i in range(config.G)]
		group_max_value = [0 for i in range(config.G)]
		for i in range(len(config.CLIENTS_LIST)):
			label = self.group_labels[i]
			if ttpi_order[i] >= group_max_value[label]:
				group_max_value[label] = ttpi_order[i]
				group_max_index[label] = i

		ttpi_order = np.array(ttpi_order)[np.array(group_max_index)]
		offloading_order = np.array(offloading_order)[np.array(group_max_index)]
		state = np.append(ttpi_order, offloading_order)
		return state

	def get_offloading(self, split_layer):
		offloading = {}
		workload = 0

		assert len(split_layer) == len(config.CLIENTS_LIST)
		for i in range(len(config.CLIENTS_LIST)):
			for l in range(len(config.model_cfg[config.model_name])):
				if l <= split_layer[i]:
					workload += config.model_cfg[config.model_name][l][5]
			offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
			workload = 0

		return offloading

		self.initialize(split_layers,  LR)

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)

