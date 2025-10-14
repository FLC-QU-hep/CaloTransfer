import torch
import torch.nn as nn

from pyro.nn import ConditionalDenseNN, DenseNN, ConditionalAutoRegressiveNN
import pyro.distributions as dist
import pyro.distributions.transforms as T
from custom_pyro import ConditionalAffineCouplingTanH


def compile_HybridTanH_model(num_blocks, num_inputs, num_cond_inputs, device):
    """ten blocks with seven coupling layers each"""
    # the latent space distribution: choosing a 2-dim Gaussian
    base_dist = dist.Normal(torch.zeros(num_inputs).to(device), torch.ones(num_inputs).to(device))

    input_dim = num_inputs
    count_bins = 8
    transforms = []
    transforms2 = []
      
    input_dim = num_inputs
    split_dim = num_inputs//2
    param_dims1 = [input_dim-split_dim, input_dim-split_dim]
    param_dims2 = [input_dim * count_bins, input_dim * count_bins, input_dim * (count_bins - 1), input_dim * count_bins]

    torch.manual_seed(42)   # important to reproduce the permutations 

    for i in range(num_blocks):
        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)

        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        hypernet = DenseNN(num_cond_inputs, [input_dim*4, input_dim*4], param_dims2)
        #hypernet.apply(init_weights)
        ctf = T.ConditionalSpline(hypernet, input_dim, count_bins)
        transforms2.append(ctf)
        transforms.append(ctf)

        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)

        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        #hypernet.apply(init_weights)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)    
        
    modules = nn.ModuleList(transforms2)

    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transforms)

    return modules, flow_dist

def add_conditional_affine_coupling(transforms, transforms2, split_dim, num_cond_inputs, param_dims1):
    hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [split_dim*10, split_dim*10], param_dims1)
    ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
    transforms2.append(ctf)
    transforms.append(ctf)

def add_permutation(transforms, input_dim, device):
    perm = torch.randperm(input_dim, dtype=torch.long).to(device)
    ff = T.Permute(perm)
    transforms.append(ff)

def compile_HybridTanH_model_CaloC2(num_blocks, num_inputs, num_cond_inputs, device):
    # the latent space distribution: choosing a 2-dim Gaussian
    base_dist = dist.Normal(torch.zeros(num_inputs).to(device), torch.ones(num_inputs).to(device))
    # print("base_dist: ", base_dist)
    # print(base_dist.shape)
    input_dim = num_inputs
    count_bins = 8
    transforms = []
    transforms2 = []
      
    split_dim = num_inputs // 2
    param_dims1 = [split_dim, split_dim]
    param_dims2 = [input_dim * count_bins, input_dim * count_bins, input_dim * (count_bins - 1), input_dim * count_bins]

    torch.manual_seed(42)

    for _ in range(num_blocks):
        for _ in range(3):
            add_conditional_affine_coupling(transforms, transforms2, split_dim, num_cond_inputs, param_dims1)
            add_permutation(transforms, input_dim, device)
        
        add_permutation(transforms, input_dim, device)
        
        hypernet = DenseNN(num_cond_inputs, [input_dim*4, input_dim*4], param_dims2)
        ctf = T.ConditionalSpline(hypernet, input_dim, count_bins)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        for _ in range(3):
            add_conditional_affine_coupling(transforms, transforms2, split_dim, num_cond_inputs, param_dims1)
            add_permutation(transforms, input_dim, device)
        
    modules = nn.ModuleList(transforms2)
    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transforms)

    return modules, flow_dist