import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from mmd import mix_rbf_mmd2
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(2 * dim, 7),
            nn.ELU(),
            nn.Linear(7, 13),
            nn.ELU(),
            nn.Linear(13, 7),
            nn.ELU(),
            nn.Linear(7, dim)
        )
        self.model = model

    def forward(self, input):
        return self.model(input)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(dim, 11),
            nn.ELU(),
            nn.Linear(11, 29),
            nn.ELU()
        )
        self.model = model

    def forward(self, input):
        return self.model(input)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(29, 11),
            nn.ELU(),
            nn.Linear(11, dim),
        )
        self.model = model

    def forward(self, input):
        return self.model(input)

def data_sampler(dist_type, dist_param, batch_size=1, cond=None):
    if cond:
        if dist_type=="gaussian":
            return Tensor(np.random.normal(cond, dist_param[1], (batch_size, dim))).requires_grad_()
        elif dist_type=="mvn":
            return Tensor(np.random.multivariate_normal(cond, dist_param[1], (batch_size))).requires_grad_()
    else:
        if dist_type=="gaussian":
            return Tensor(np.random.normal(dist_param[0], dist_param[1], (batch_size, dim))).requires_grad_()
        elif dist_type=="uniform":
            return Tensor(np.random.uniform(dist_param[0], dist_param[1], (batch_size, dim))).requires_grad_()
        elif dist_type=="cauchy":
            return dist_param[1] * Tensor(np.random.standard_cauchy((batch_size, dim))) + 23.
        elif dist_type=="mvn":
            return Tensor(np.random.multivariate_normal(dist_param[0], dist_param[1], (batch_size))).requires_grad_()

# hyper parameters
num_iteration = 10000
num_gen = 1
num_enc_dec = 5
lr = 1e-3
batch_size = 128
target_dist = "mvn"
mean = [[-1, 1], [1, -1]]
cov = [[1, 0.5], [0.5, 1]]
target_param = (None, cov)
dim = 2
noise_dist = "uniform"
noise_param = (-1, 1)

# MMD parameters
lambda_AE = 8.
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]
print_int = 1000

gen = GNet()
enc = Encoder()
dec = Decoder()
criterion = nn.MSELoss()
gen_optimizer = optim.Adam(gen.parameters(), lr=lr)
enc_optimizer = optim.Adam(enc.parameters(), lr=lr)
dec_optimizer = optim.Adam(dec.parameters(), lr=lr)
cum_dis_loss = 0
cum_gen_loss = 0

for iteration in range(num_iteration):
    idx = np.random.randint(0, 2)
    for i in range(num_enc_dec):
        enc.zero_grad()
        dec.zero_grad()
        target = data_sampler(target_dist, target_param, batch_size, mean[idx])
        noise = data_sampler(noise_dist, noise_param, batch_size)
        encoded_target = enc.forward(target)
        decoded_target = dec.forward(encoded_target)
        L2_AE_target = (target - decoded_target).pow(2).mean()
        
        new_noise = torch.cat((noise, Tensor(mean[idx]).expand(batch_size, 2)), 1)
        transformed_noise = gen.forward(new_noise)
        encoded_noise = enc.forward(transformed_noise)
        decoded_noise = dec.forward(encoded_noise)
        L2_AE_noise = (transformed_noise - decoded_noise).pow(2).mean()
        MMD = mix_rbf_mmd2(encoded_target, encoded_noise, sigma_list)
        MMD = F.relu(MMD)
        L_MMD_AE = -1 * (torch.sqrt(MMD)-lambda_AE*(L2_AE_noise+L2_AE_target))
        L_MMD_AE.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        cum_dis_loss = cum_dis_loss - L_MMD_AE.item()
    for i in range(num_gen):
        gen.zero_grad()
        target = data_sampler(target_dist, target_param, batch_size, mean[idx])
        noise = data_sampler(noise_dist, noise_param, batch_size)
        new_noise = torch.cat((noise, Tensor(mean[idx]).expand(batch_size, 2)), 1)
        encoded_target = enc.forward(target)
        encoded_noise = enc.forward(gen.forward(new_noise))
        MMD = torch.sqrt(F.relu(mix_rbf_mmd2(encoded_target, encoded_noise, sigma_list)))
        MMD.backward()
        gen_optimizer.step()
        cum_gen_loss = cum_gen_loss + MMD.item()
    if iteration % print_int == 0 and iteration != 0:
        print("iteration:", iteration)
        print('cum_dis_loss {}, cum_gen_loss {}'.format(cum_dis_loss/(print_int*num_enc_dec), cum_gen_loss/(print_int*num_gen)))
        cum_dis_loss = 0
        cum_gen_loss = 0

# In-Distribution Plot
samples = data_sampler(target_dist, target_param, 100000, mean[0]).transpose(0, 1).detach().numpy()
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
axs[0].hist2d(samples[0], samples[1], bins=(50, 50), cmap=plt.cm.jet)
axs[0].set_title('True Distribution')

noise = data_sampler(noise_dist, noise_param, 100000)
new_noise = torch.cat((noise, Tensor(mean[0]).expand(100000, 2)), 1)
transformed_noise = gen.forward(new_noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000, dim).transpose(1, 0)
axs[1].hist2d(transformed_noise[0], transformed_noise[1], bins=(50, 50), cmap=plt.cm.jet)
axs[1].set_title('Approximate Distribution')
plt.show()

# Out of Distribution Plot
samples = data_sampler(target_dist, target_param, 100000, (0.5, 0.5)).transpose(0, 1).detach().numpy()
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
axs[0].hist2d(samples[0], samples[1], bins=(50, 50), cmap=plt.cm.jet)
axs[0].set_title('True Distribution')

noise = data_sampler(noise_dist, noise_param, 100000)
new_noise = torch.cat((noise, Tensor((0.5, 0.5)).expand(100000, 2)), 1)
transformed_noise = gen.forward(new_noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000, dim).transpose(1, 0)
axs[1].hist2d(transformed_noise[0], transformed_noise[1], bins=(50, 50), cmap=plt.cm.jet)
axs[1].set_title('Approximate Distribution')
plt.show()
