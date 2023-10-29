
# Communicator Object

import pickle
import struct
import socket


import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Communicator(object):
	def __init__(self, index, ip_address):
		self.index = index
		self.key = ip_address
		self.sock = socket.socket()


	def send_msg(self, sock, msg):
		msg_pickle = pickle.dumps(msg)
		sock.sendall(struct.pack(">I", len(msg_pickle)))
		sock.sendall(msg_pickle)
		logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	def recv_msg(self, sock, expect_msg_type=None):
		msg_len = struct.unpack(">I", sock.recv(4))[0]
		msg = sock.recv(msg_len, socket.MSG_WAITALL)
		msg = pickle.loads(msg)
		logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

		if expect_msg_type is not None:
			if msg[0] == 'Finish':
				return msg
			elif msg[0] != expect_msg_type:
				raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
		return msg


# 导包
import socket
import pickle


def socket_tcp_recv(sock,buffer_size):  
    all_data_received_flag = False
    received_data = b""
    while True:
        try:
            data = sock.recv(buffer_size)
            received_data += data

            try:  
                pickle.loads(received_data)
                # If the previous pickle.loads() statement is passed, this means all the data is received.
                # Thus, no need to continue the loop. The flag all_data_received_flag is set to True to signal all data is received.
                all_data_received_flag = True
            except BaseException:  # pickle data was truncated
                # An exception is expected when the data is not 100% received.
                pass

            if data == b'':  # Nothing received from the client. 
                received_data = b""
                # If still nothing received for a number of seconds specified by the socket_tcp_recv_timeout attribute, return with status 0 to close the connection.
                # if (time.time() - self.socket_tcp_recv_start_time) > self.socket_tcp_recv_timeout: 
                #     return None, 0  # 0 means the connection is no longer active and it should be closed.

            elif all_data_received_flag: 
                # print(
                #     "All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info,
                #                                                                       data_len=len(received_data)))

                if len(received_data) > 0:  
                    try:
                        # Decoding the data (bytes).  
                        # received_data = pickle.loads(received_data)
                        # Returning the decoded data.
                        return received_data

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                        return None

            # else:
            #     In case data are received from the client, update the socket_tcp_recv_start_time to the current time to reset the timeout counter.
                # self.socket_tcp_recv_start_time = time.time()

        except BaseException as e:
            print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
            return None


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

def recv_index_from_server(server_addr):

    client_socket.connect(server_addr)

    msg_bytes = socket_tcp_recv(client_socket,1024)
    msg = pickle.loads(msg_bytes)

    print("key and index received:", msg)
    
    # client_socket.close()
    return client_socket,msg['client_key'],msg['index'],msg['iteration']
    
def get_clients_info(client_num,server_addr):
    print("get client info-------")
    self_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    self_sock.bind(server_addr)

    client_sockets_dict = {} # “IP:port" --> socket_obj 

    clients_key_list = []  # client_key : "IP:port"

    self_sock.listen(client_num+10)
    while len(client_sockets_dict) < client_num:

        client_socket, (client_ip, client_port) = self_sock.accept()
        print('Got connection from:'+ str(client_ip) + ":"+ str(client_port))
        
        client_key = str(client_ip)+":"+str(client_port)
        clients_key_list.append(client_key)
        client_sockets_dict[client_key] = client_socket

        print('len(client_sockets):',len(client_sockets_dict))

    return clients_key_list,client_sockets_dict,self_sock

def send_index_to_clients(client_sockets_dict):
    print("------send_index_to_clients-----")
    index = 0
    iteration = 5
    for key,client_sock in client_sockets_dict.items():
        msg = {'client_key':key,'index':index,'iteration':iteration}
        msg_pickle = pickle.dumps(msg)
        index += 1
        client_sock.sendall(msg_pickle)
