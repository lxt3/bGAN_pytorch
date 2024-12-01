import torch
import torch.nn as nn

class _netG64(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3):
        super(_netG64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netG(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, batch_norm_layers=[], affine=True):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        if batch_norm_layers == []:
            print("Initializing the Batch Norm layers. Affine = {}".format(affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 8, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 4, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 2, affine=affine))
        else:
            assert len(batch_norm_layers) == 3
            #print("Reusing the Batch Norm Layers")
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            batch_norm_layers[0],
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            batch_norm_layers[1],
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            batch_norm_layers[2],
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class _netG_v2(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, batch_norm_layers=[], affine=True):
        super(_netG_v2, self).__init__()
        self.ngpu = ngpu
        if batch_norm_layers == []:
            print("Initializing the Batch Norm layers. Affine = {}".format(affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 8, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 4, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 2, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 1, affine=affine))
        else:
            assert len(batch_norm_layers) == 4
            print("Reusing the Batch Norm Layers")
        self.main = nn.Sequential(
            # input is Z, going into a a linear layer and reshape
            Reshape(-1, nz),
            nn.Linear(in_features=nz, out_features=2*2*ngf*8, bias=True),
            Reshape(-1, ngf*8, 2, 2),
            #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            # in: 2 x 2 with kernel 5, stride 2, padding 2
            batch_norm_layers[0],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*8, ngf * 4, kernel_size=4, stride=2, padding=1),
            batch_norm_layers[1],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*4, ngf * 2, 4, stride=2, padding=1),
            batch_norm_layers[2],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*2, ngf * 1, 4, stride=2, padding=1),
            batch_norm_layers[3],
            nn.ReLU(True),
            nn.ConvTranspose2d(  ngf*1,      nc, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netG_synth(nn.Module):
    def __init__(self, ngpu, nz=2, dimx=100, leaky_inplace=False):
        super(_netG_synth, self).__init__()
        self.ngpu = ngpu
        # map 2 dim to 100 dim
        self.main = nn.Sequential(
            nn.Linear(nz, 1000, bias=True),
            nn.LeakyReLU(0.2, inplace=leaky_inplace),
            nn.Linear(1000, dimx)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

    
    
#     def __init__(self, ngpu, nz, dimx, threshold_prob, binary_indices=[], leaky_inplace=False, batch_norm_layers=[], affine=True):    
class _netG_ess(nn.Module):
    def __init__(self, ngpu, nz, dimx, binary_indices=[], leaky_inplace=False, batch_norm_layers=[], affine=True):
        super(_netG_ess, self).__init__()
        self.ngpu = ngpu
        self.binary_indices = binary_indices  # Indices of binary variables
        self.threshold_prob = nn.Parameter(torch.Tensor([0.5]))  # Adjust initial value if needed
        
        self.sigmoid = nn.Sigmoid()
        #self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=8)  # Adjust dimensions as needed
        
        if batch_norm_layers == []:
            print("Initializing the Batch Norm layers. Affine = {}".format(affine))
            batch_norm_layers.append(nn.BatchNorm1d(1000, affine=affine))
            batch_norm_layers.append(nn.BatchNorm1d(500, affine=affine))
            batch_norm_layers.append(nn.BatchNorm1d(250, affine=affine))
        else:
            assert len(batch_norm_layers) == 3
            #print("Reusing the Batch Norm Layers")

        # map nz dim to x dim
        self.main = nn.Sequential(
            nn.Linear(nz, 1000, bias=True),
#            batch_norm_layers[0],
            nn.LeakyReLU(0.2, inplace=leaky_inplace),
            nn.Linear(1000, dimx-1)      
        )
        
        self.output_binary = nn.Sequential(
            nn.Linear(nz, 1000, bias=True),
            #nn.LeakyReLU(0.2, inplace=leaky_inplace),
            nn.Linear(1000, 1),  # Specifically for the binary feature
            nn.BatchNorm1d(1),   # Normalize binary feature output prior to sigmoid
            nn.Sigmoid()         # Apply sigmoid
        )        
        

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:         
            #output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            features = nn.parallel.data_parallel(self.main, input, range(self.ngpu))  # torch.Size([64, 38])            
            binary_prob = nn.parallel.data_parallel(self.output_binary, input, range(self.ngpu))
            binary_feature = (binary_prob > self.threshold_prob).float()
#            binary_feature = nn.parallel.data_parallel(self.output_binary, input, range(self.ngpu))
            output = torch.cat((features, binary_feature), dim=1)
        else:
            #output = self.main(input)     
            features = self.main(input)
            binary_prob = self.output_binary(input)  # Probability from sigmoid for binary feature
            # Apply threshold to get binary output
            binary_feature = (binary_prob > self.threshold_prob).float()            
            #binary_feature = self.output_binary(input)
            output = torch.cat((features, binary_feature), dim=1)        # Apply sigmoid to the specified binary dimensions
        #output[:, self.binary_indices] = self.sigmoid(output[:, self.binary_indices])
        
        return output


    
    

class _netG_ess_cond(nn.Module):
    def __init__(self, ngpu, n_classes, embedding_dim, nz, dimx, binary_indices=[], leaky_inplace=False, threshold_prob=0.5, batch_norm_layers=[], affine=True):
        super(_netG_ess_cond, self).__init__()
        self.ngpu = ngpu
        self.binary_indices = binary_indices  # Indices of binary response variables (not binary conditions)
        
        self.sigmoid = nn.Sigmoid()
        self.threshold_prob = threshold_prob
        
        if batch_norm_layers == []:
            print("Initializing the Batch Norm layers. Affine = {}".format(affine))
            batch_norm_layers.append(nn.BatchNorm1d(1000, affine=affine))
            batch_norm_layers.append(nn.BatchNorm1d(500, affine=affine))
            batch_norm_layers.append(nn.BatchNorm1d(250, affine=affine))
        else:
            assert len(batch_norm_layers) == 3

        # Label-conditioned generator component
        self.label_conditioned_generator = nn.Sequential(
            nn.Embedding(n_classes, embedding_dim),
            nn.Linear(embedding_dim, 32),
            nn.ReLU()
        )
        # Latent space to hidden layer
        self.latent = nn.Sequential(
            nn.Linear(nz, 128),
            nn.ReLU()
        )

        # map nz dim + class dim to x dim
        self.main = nn.Sequential(
            nn.Linear(32+128, 1000, bias=True),
        #    batch_norm_layers[0],
            nn.LeakyReLU(0.2, inplace=leaky_inplace),
            nn.Linear(1000, dimx)
        )


    def forward(self, noise_vector, label):
        label_output = self.label_conditioned_generator(label)
        latent_output = self.latent(noise_vector)
        # Concatenating label conditioned output and latent output from generator
        concat = torch.cat([latent_output, label_output], dim=1)    

        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, concat, range(self.ngpu))
        else:
            output = self.main(concat)
        # Apply sigmoid to the specified binary dimensions
        output[:, self.binary_indices] = self.sigmoid(output[:, self.binary_indices])
        output[:, self.binary_indices] = (output[:, self.binary_indices] > self.threshold_prob).float() # death
        return output