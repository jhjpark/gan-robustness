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
            nn.Linear(2 * dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, dim)
        )
        self.model = model

    def forward(self, input):
        return self.model(input)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU()
        )
        self.model = model

    def forward(self, input):
        return self.model(input)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, dim),
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

# data
im_height = 8
im_width = 8
new_img = x_train[0][:im_height, :im_width, :]
mean = list(new_img.flatten())
cov = torch.zeros((im_height * im_width * 3, im_height * im_width * 3))
for i in range(3 * im_height * im_width):
    for j in range(3 * im_height * im_width):
        if i == j:
            cov[i][j] = 50
        else:
            cov[i][j] = 25
            cov[i][j] = 25
plt.imshow(Tensor(mean).reshape((im_height, im_width, 3)).numpy().astype("uint8"))
plt.show()

# hyper parameters
num_iteration = 5000
num_gen = 1
num_enc_dec = 5
lr = 1e-3
batch_size = 128
target_dist = "mvn"
target_param = (None, cov)
dim = im_height * im_width * 3
noise_dist = "uniform"
noise_param = (-1, 1)

# MMD parameters
lambda_AE = 8.
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]
print_int = 100

gen = GNet()
enc = Encoder()
dec = Decoder()
criterion = nn.MSELoss()
gen_optimizer = optim.Adam(gen.parameters(), lr=lr)
enc_optimizer = optim.Adam(enc.parameters(), lr=lr)
dec_optimizer = optim.Adam(dec.parameters(), lr=lr)
cum_dis_loss = 0
cum_gen_loss = 0

plt.imshow(data_sampler(target_dist, target_param, 1, mean).detach().numpy().reshape(1, dim).transpose(1, 0).reshape((im_height, im_width, 3)).astype("uint8"))

for iteration in range(num_iteration):
    for i in range(num_enc_dec):
        enc.zero_grad()
        dec.zero_grad()
        target = data_sampler(target_dist, target_param, batch_size, mean)
        noise = data_sampler(noise_dist, noise_param, batch_size)
        encoded_target = enc.forward(torch.cat((target, Tensor(mean).expand(batch_size, dim)), 1))
        decoded_target = dec.forward(encoded_target)
        L2_AE_target = (target - decoded_target).pow(2).mean()
        
        new_noise = torch.cat((noise, Tensor(mean).expand(batch_size, dim)), 1)
        transformed_noise = gen.forward(new_noise)
        encoded_noise = enc.forward(torch.cat((transformed_noise, Tensor(mean).expand(batch_size, dim)), 1))
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
        target = data_sampler(target_dist, target_param, batch_size, mean)
        noise = data_sampler(noise_dist, noise_param, batch_size)
        new_noise = torch.cat((noise, Tensor(mean).expand(batch_size, dim)), 1)
        encoded_target = enc.forward(torch.cat((target, Tensor(mean).expand(batch_size, dim)), 1))
        encoded_noise = enc.forward(torch.cat((gen.forward(new_noise), Tensor(mean).expand(batch_size, dim)), 1))
        MMD = torch.sqrt(F.relu(mix_rbf_mmd2(encoded_target, encoded_noise, sigma_list)))
        MMD.backward()
        gen_optimizer.step()
        cum_gen_loss = cum_gen_loss + MMD.item()
    if iteration % print_int == 0 and iteration != 0:
        print("iteration:", iteration)
        print('cum_dis_loss {}, cum_gen_loss {}'.format(cum_dis_loss/(print_int*num_enc_dec), cum_gen_loss/(print_int*num_gen)))
        cum_dis_loss = 0
        cum_gen_loss = 0

        # plot
        noise = data_sampler(noise_dist, noise_param, 1)
        new_noise = torch.cat((noise, Tensor(mean).expand(1, dim)), 1)
        transformed_noise = gen.forward(new_noise)
        transformed_noise = transformed_noise.data.numpy().reshape(1, dim).transpose(1, 0)
        transformed_noise = transformed_noise.reshape((im_height, im_width, 3)).astype("uint8")
        plt.imshow(transformed_noise)
        plt.show()
