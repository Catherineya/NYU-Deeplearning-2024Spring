import torch
class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function
        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        #TODO: Implement the forward function
        #Linear 1
        self.cache['x'] = x
        s_1 = x @ self.parameters['W1'].T + self.parameters['b1'] 
        self.cache['s_1'] = s_1
        
        # function f
        if self.f_function == 'identity':
            a_1 = s_1
        elif self.f_function == 'relu':
            a_1 = torch.relu(s_1)
        elif self.f_function == 'sigmoid':
            a_1 = torch.sigmoid(s_1)
            
        self.cache['a_1'] = a_1
            
        #Linear 2
        s_2 = a_1 @ self.parameters['W2'].T + self.parameters['b2']
        self.cache['s_2'] = s_2
        
        # function g
        if self.g_function == 'identity':
            y_hat = s_2
        elif self.g_function == 'relu':
            y_hat = torch.relu(s_2)
        elif self.g_function == 'sigmoid':
            y_hat = torch.sigmoid(s_2)
            
        self.cache['y_hat'] = y_hat
        # print("y_hat shape:", y_hat.shape)
        
        return y_hat
        

    
 
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        #print("dJdy_hat shape:", dJdy_hat.shape)
        # function g
        if self.g_function == 'identity':
            dJds_2 = dJdy_hat
        elif self.g_function == 'relu':
            dJds_2 = dJdy_hat * (self.cache['s_2'] > 0).float()
        elif self.g_function == 'sigmoid':
            dJds_2 = dJdy_hat * torch.sigmoid(self.cache['s_2']) * (1 - torch.sigmoid(self.cache['s_2']))
            
        #Linear 2
        self.grads['dJdW2'] = dJds_2.T @ self.cache['a_1']
        # self.grads['dJdb2'] = dJds_2.sum()
        self.grads['dJdb2']=dJds_2.T @ torch.ones(dJds_2.shape[0])
        
        # function f
        if self.f_function == 'identity':
            dJda_1 = dJds_2 @ self.parameters['W2']
        elif self.f_function == 'relu':
            dJda_1 = dJds_2 @ self.parameters['W2'] * (self.cache['s_1'] > 0).float()
        elif self.f_function == 'sigmoid':
            dJda_1 = dJds_2 @ self.parameters['W2'] * torch.sigmoid(self.cache['s_1']) * (1 - torch.sigmoid(self.cache['s_1']))
        
        #Linear 1
        self.grads['dJdW1'] = dJda_1.T @ self.cache['x']
        # self.grads['dJdb1'] = dJda_1.sum()
        self.grads['dJdb1'] = dJda_1.T @ torch.ones(dJda_1.shape[0])
        
        
    
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = torch.mean((y_hat - y)**2)
    dJdy_hat = 2 * (y_hat - y) / (y.shape[0] * y.shape[1])
    return loss, dJdy_hat


    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat)) / (y.shape[0] * y.shape[1])
    return loss, dJdy_hat











