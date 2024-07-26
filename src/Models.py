import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module
    
    It's standard to instantiate custom activation function as child classes
    of nn.Module
    """
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class NaisNet(nn.Module):

    def __init__(self, layers, stable, activation):
        """ 
        This is formulated differently from original Repo
        Original repo: hard-coded to be 5 layers
        Parpas' version: allows for few layers - but in layers = 6 to have same number of layers as Original repo
        """
        super(Naisnet, self).__init__()

        self.layers = layers

        # This build out the layers in ResNet
        self.layer1 = nn.Linear(in_features=layers[0], out_features=layers[1])

        self.layer2 = nn.Linear(in_features=layers[1], out_features=layers[2])
        self.layer2_input = nn.Linear(in_features=layers[0], out_features=layers[2])
        
        self.layer3 = nn.Linear(in_features=layers[2], out_features=layers[3])
        if len(layers) == 5:
            self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
            self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
        elif len(layers) == 6:
            self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
            self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
            self.layer4_input = nn.Linear(in_features=layers[0], out_features=layers[4])
            self.layer5 = nn.Linear(in_features=layers[4], out_features=layers[5])

        self.activation = activation

        self.epsilon = 0.01
        self.stable = stable

    def project(self, layer, out):  # Building block for the NAIS-Net
        """ 
        This is the equivalent to stable_forward in the Original Repo

        ## Notes on NAIS-Net implementation:
        Recall NAIS-Net is based on blocks of:
            x(k+1) = x(k) + h*f(Ax(k) + Bu + b)
        
        where A = -R~*R~ - epsilon*I
        and R~ is based on projection of R using 'delta' & Frobenius norm.
        The below code seems to otput the below linear transformation, which
        is used within the activation function:
            Ax(k) + b
        
        The reset of the NAIS-Net layer-connection (activation function, identity connection)
        will be built in the NaisNet.foward method


        ## Notes on PyTorch implimentation:
        It's standard for nn.Modules to define a 'forward' method 
        which indicates the forward pass used in Neural Network layers
        i.e. what each layer will do to the input data.

        Although this .project method outputs a LINEAR transformation,
        it is THEN used in the main NaisNet.forward method, which will wrap this linear
        transformation WITHIN a non-linear activation function AND add in the identity
        from the past layer to make it like ResNet
        """
        weights = layer.weight
        
        # NAIS-Net Paper: the below actions the algorithm to achieve condition 1, in Section 4
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(weights.t(), weights)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** (1 / 2) * RtR / (norm ** (1 / 2))
        
        # The below actions -R~*R~ - epsilon*I
        # torch.eye creates identity matrix of given dimensions
        # A = RtR + torch.eye(RtR.shape[0]).cuda() * self.epsilon ## => Parpas takes out the .cuda() tail!
        A = RtR + torch.eye(RtR.shape[0]) * self.epsilon

        # The below indicates that NaisNet.project will apply the above linear transformation
        # on incoming data
        # F.linear(x,y,bias) => y = xW + bias
        return F.linear(out, -A, layer.bias)

    def forward(self, x):
        u = x

        # Here, we pass the layer's input through the layer1 transformation
        # (which is just linear)
        # and then feed this into whatever activation function we have chosen
        # This is more like a standard NN layer design

        out = self.layer1(x)
        out = self.activation(out)

        shortcut = out

        if self.stable:
            # Pass through the .project logic which gives 'Ax(k) + b'
            out = self.project(self.layer2, out)
            # Here, we add in the 'Bu' to layer2 pass so we now have 'Ax(k) + Bu + b
            out = out + self.layer2_input(u)

        else:
            out = self.layer2(out)
        
        # We then wrap in actiavtion function: 'f(Ax(k) + Bu + b)'
        out = self.activation(out)
        # Finally, we add the identity component: 'x(k) + f(Ax(k) + Bu + b)'
        out = out + shortcut

        # If we limit our layers to 4, then we do not need to do any further NAIS-Net connections.
        # We feed into the final layer and return
        if len(self.layers) == 4:
            # In this case, this is our output layer, so we just want to output a scalar - this will achieve that
            out = self.layer3(out)
            return out

        # If we continue with further layers, we do the exact same process as before
        if len(self.layers) == 5:
            # We update our shortcut to output of the layer, for use in subsequent layers
            shortcut = out

            # Same as before, for layer 3
            if self.stable:
                out = self.project(self.layer3, out)
                out = out + self.layer3_input(u)
            else:
                out = self.layer3(out)
            out = self.activation(out)
            out = out + shortcut

            out = self.layer4(out)
            return out
        
        if len(self.layers) == 6:
            shortcut = out
            if self.stable:
                out = self.project(self.layer3, out)
                out = out + self.layer3_input(u)
            else:
                out = self.layer3(out)
            out = self.activation(out)
            out = out + shortcut


            shortcut = out
            if self.stable:
                out = self.project(self.layer4, out)
                out = out + self.layer4_input(u)
            else:
                out = self.layer4(out)

            out = self.activation(out)
            out = out + shortcut

            out = self.layer5(out)
            return out

        return out

