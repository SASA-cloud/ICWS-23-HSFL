import numpy as np

def fillThetaContext(layerInfo, theta_context_dim): 
    Action_num = len(layerInfo)
    x_theta = np.zeros((theta_context_dim, Action_num))
    for i in range(Action_num): 
        x_theta[0][i] = layerInfo[i][0]
        x_theta[1][i] = layerInfo[i][1]
        # actionList.append(layerInfo[i][7]) # partition point
    return x_theta


class muLinUCB():
    def __init__(self, layerInfo, privacyLeakage,actionList):
        # self.mu = mu
        self.numOfAction = len(layerInfo)
        self.thetaContextDim = 2 
        self.x_theta = fillThetaContext(layerInfo, self.thetaContextDim) 
        self.actionList = actionList


        self.privacyLeakage = privacyLeakage

        self.A = np.diag(np.random.randint(1, 9, size=self.thetaContextDim))
        self.b = np.zeros((self.thetaContextDim, 1))

        self.alpha = 0.25
        self.latency_weight = 1.0

    def getEstimationAction(self):
        A_inv = np.linalg.inv(self.A) 
        theta = np.matmul(A_inv, self.b)

        training_cost = []

        for action_index in range(self.numOfAction):
            x_1 = np.copy(self.x_theta[:, [action_index]]) 
            x_2 = np.copy(self.x_theta[:, [action_index]])

            temp_1 = np.matmul(x_1.T, theta) 
            temp_2 = self.alpha * np.sqrt(np.matmul(np.matmul(x_1.T, A_inv), x_2))

            training_cost.append(self.latency_weight*(temp_1 - temp_2) + (1.0-self.latency_weight)*self.privacyLeakage[action_index]) # trade off 

        estimate_action_index = training_cost.index(min(training_cost))
        estimate_action = self.actionList[estimate_action_index]
        return estimate_action

    def updateA_b(self, estimate_action, actual_delay):
        estimate_action_index = self.actionList.index(estimate_action)
        self.A = self.A + np.matmul(self.x_theta[:, [estimate_action_index]], self.x_theta[:, [estimate_action_index]].T)
        self.b = self.b + self.x_theta[:, [estimate_action_index]] * actual_delay












