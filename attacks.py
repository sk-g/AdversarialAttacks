import torch
from torch import nn
from torch.autograd import Variable


def set_grad(model, cond):
    """Set type of parameter,
    to compute grad or ignore
    """
    for p in model.parameters():
        p.requires_grad = cond

class Attack(object):

    def __init__(self,model=None,criterion=None,cuda = False):
        self.model = model
        self.criterion = criterion
        if not cuda:
            self.device = torch.device('cpu')
        else:
            # force cou as default device if cuda is unavailable
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        self.model.to(self.device)


    def generate_example(self, seed_img, attack, target=None, ground=None, **kwargs):
        """
        """
        set_grad(self.model, False)
        img_var = Variable(seed_img.clone(), requires_grad=True).to(self.device)
        
        if attack == 'GA':
            res = self.GA(img_var, target, ground, **kwargs)
        elif attack == 'FGS':
            res = self.FGSM(img_var, target, ground, **kwargs)
        else:
            raise Exception('{} attack not implemented'.format(attack))
        
        set_grad(self.model, True)
        if type(res) == tuple:
            fool_image, iters = res
            return fool_image.data.cpu(), iters
        else:
            fool_image = res
            return fool_image.data.cpu()    
    def FGSM(self, img_var, target=None, 
                ground=None, 
                alpha = 1,
                n_iters=1, 
                eta=0.007, 
                stop_early=True):
        """Function to perform 
        fast gradient sign method (FGSM) and iterative FGSM
        """
        x = img_var
        if ground is None:
            try:
                pred = self.model(img_var)
                pred = pred.max(0)[1].data.item()
                ground = pred
            except:
                raise Exception('true label not provided : {}'.format(None))
        x_adv = Variable(x.data, requires_grad=True).to(self.device)
        # print("FGSM function x_adv shape:",x_adv.shape) ->correct shape
        # try
        print(self.model)
        for i in range(n_iters):
            h_adv = self.model(x_adv)
            if target is not None:
                # target attack
                cost = self.criterion(h_adv,ground)
            else:
                cost = -self.criterion(h_adv,y)
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            #inplace sign of grads
            x_adv.grad.sign_()
            #perturb based on opp direction of sign
            x_adv = x_adv - alpha*x_adv,grad
            #clip values so that added noise is controlled
            x_adv = where(x_adv > x+eps, x+eps, x_adv)
            x_adv = where(x_adv < x-eps, x-eps, x_adv)
            x_adv = Variable(x_adv.data,requires_grad = True)
        print(x.shape,x_adv.shape)        
        # return img_var, i    