import torch 
from torch.autograd import grad

class LinearRegression():
    """
      Linear Regression with manual backward propagation
    """
    def __init__(self, n_features=2, lr=1e-3):
        self.n_features = n_features
        self.w = torch.zeros(self.n_features, 1, dtype=torch.float64)
        self.b = torch.ones(1, dtype=torch.float64)
        self.lr = lr
        
    def activation_func(self,x):
        return x
    
    def forward(self, x):
        z = x @ self.w + self.b
        a = self.activation_func(z)
        return a #.view(-1)
    
    def backward(self, x, y, y_hat):
      """
        Manual backward propagation
      """
        g_loss_wrt_yhat  = 2 * (y - y_hat)
        g_yhat_wrt_w = -x
        g_yhat_wrt_b = -1
        
        g_loss_wrt_w = g_yhat_wrt_w.t() @ g_loss_wrt_yhat / y.size(0) #.view(-1, 1)
        g_loss_wrt_b = torch.sum(g_loss_wrt_yhat * g_yhat_wrt_b) / y.size(0)
        
        return (-1)*g_loss_wrt_w, (-1)*g_loss_wrt_b
    
    def optimize(self, g_w, g_b):
        self.w = self.w + self.lr * g_w
        self.b = self.b + self.lr * g_b
        
class LinearRegression1():
    """
      Linear Regression with semi manual backward propagation
      using grad function
    """
    
    def __init__(self, n_features=2, lr=1e-3):
        self.n_features = n_features
        self.w = torch.zeros(self.n_features, 1, dtype=torch.float64,
                            requires_grad=True)
        self.b = torch.ones(1, dtype=torch.float64,
                            requires_grad=True)
        self.lr = lr
        
    def activation_func(self,x):
        return x
    
    def forward(self, x):
        z = x @ self.w + self.b
        a = self.activation_func(z)
        return a #.view(-1)
    
    def backward(self, loss):
        """
          Semi manual backward propagation
        """
        g_loss_wrt_w = grad(loss, self.w, retain_graph=True)[0] #.view(-1, 1)
        g_loss_wrt_b = grad(loss, self.b)[0]
        
        return (-1)*g_loss_wrt_w, (-1)*g_loss_wrt_b
    
    def optimize(self, g_w, g_b):
        self.w = self.w + self.lr * g_w
        self.b = self.b + self.lr * g_b
        



