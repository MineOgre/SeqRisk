import ast
import os, sys, torch, argparse
import time

import pandas as pd
from lifelines import CoxPHFitter
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math
import wandb
from torch.nn import Linear, TransformerEncoderLayer, TransformerEncoder

from predict_HealthMNIST import VAEoutput
from dataset_def import HealthMNISTDatasetConv, RotatedMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDataset, \
    PhysionetDataset, HUSCorogeneDataset
from parse_model_args import VAEArgs
from utils import cox_loss_last_n, HensmanDataLoader, VaryingLengthBatchSampler, VaryingLengthSubjectSampler, \
    calculate_risk_score


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.context_vector = nn.Parameter(torch.rand(feature_dim, 1), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)  # Apply softmax across the sequence_length dimension

    def forward(self, x, mask=None):
        # x: Transformer encoder output with shape [batch_size, sequence_length, features]
        # mask: Boolean tensor with shape [batch_size, sequence_length] where True indicates a padding position

        # Compute raw attention scores by dotting the context vector with each encoder output
        attention_scores = torch.matmul(x, self.context_vector).squeeze(-1)  # Shape [batch_size, sequence_length]

        if mask is not None:
            # Set attention scores at padding positions to a large negative value, ensuring they do not contribute to the softmax
            attention_scores = attention_scores.masked_fill((mask), float('-inf'))

        # Compute attention weights using softmax
        attention_weights = self.softmax(attention_scores)  # Shape [batch_size, sequence_length]

        # Compute the weighted sum of encoder outputs to get a single vector per sequence
        weighted_sum = torch.bmm(attention_weights.T.unsqueeze(1), x.transpose(1, 0)).squeeze(
            1)  # Shape [batch_size, features]

        return weighted_sum


# from model_test import VAEtest

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="LongitudinalSurvivalVAE",
#
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN-VAE",
#         "dataset": "HealthMNIST",
#         "epochs": 250,
#     }
# )
def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
    Generates a mask tensor for use in calculating conditional probabilities, where the mask's size is [N, num_Event, num_Category].
    The mask is filled with 1's up until the last measurement time for each event and category.

    Parameters:
    - meas_time: A tensor containing measurement times for each sample. Shape: [N, 1] where N is the number of samples.
    - num_Event: The number of events.
    - num_Category: The number of categories.

    Returns:
    - A mask tensor of shape [N, num_Event, num_Category] with 1's up until the last measurement time.
    '''

    N = meas_time.size(0)  # Number of samples
    mask = torch.zeros(N, num_Event, num_Category)  # Initialize mask tensor with zeros

    for i in range(N):
        # For each sample, fill with 1's up until the last measurement time
        # Note: The "+1" in the index is to include the measurement time itself as part of the mask.
        # PyTorch does not support indexing with float, so we ensure the index is an integer.
        last_meas_index = int(meas_time[i, 0].item()) + 1
        mask[i, :, :last_meas_index] = 1

    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
    Creates a mask tensor for calculating log-likelihood loss, with size [N, num_Event, num_Category].
    - If the data point is not censored, only one element corresponding to the event time will be 1 (and 0 elsewhere).
    - If the data point is censored, elements after the censoring time will be set to 1 (for all events).

    Parameters:
    - time: A tensor of times, shape [N, 1], where N is the number of samples.
    - label: A tensor of labels, shape [N, 1], where a value of 0 indicates censored data.
    - num_Event: The number of different events.
    - num_Category: The number of time categories.

    Returns:
    - A mask tensor of shape [N, num_Event, num_Category] for use in log-likelihood loss calculation.
    '''

    N = time.size(0)  # Number of samples
    mask = torch.zeros(N, num_Event, num_Category)  # Initialize the mask tensor

    for i in range(N):
        if label[i, 0] != 0:  # Not censored
            event_idx = int(label[i, 0].item()) - 1  # Event index (adjusted for 0-based indexing)
            time_idx = int(time[i, 0].item())  # time index
            mask[i, event_idx, time_idx] = 1
        else:  # Censored
            censoring_idx = int(time[i, 0].item()) + 1  # Start index for filling 1's after censoring time
            mask[i, :, censoring_idx:] = 1  # Fill with 1's after the censoring time

    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category])  # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])  # last measurement time
            t2 = int(time[i, 0])  # censoring/event time
            mask[i, (t1 + 1):(t2 + 1)] = 1  #this excludes the last measurement time and includes the event time
    else:  #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])  # censoring/event time
            mask[i, :(t + 1)] = 1  #this excludes the last measurement time and includes the event time
    return mask


def attention(query, key, value, mask=None):
    # Calculate the attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float))

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1) == True, float('-inf'))  # Use a large negative number

    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Compute the weighted sum of values
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class TransformerEncoderLayer_mine(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayer_mine, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # 1. Self-attention

        if torch.isnan(src).any():
            print("NaN values found in input to the attention module")

        src2 = attention(src, src, src, mask=src_key_padding_mask)[0]

        # src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask.T)[0]
        # print("Post Self-Attention Mean:", src2.mean().item())

        # 2. Add & Norm 1
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # print("Post Add & Norm 1 Mean:", src.mean().item())

        # 3. Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # print("Post Feedforward Mean:", src2.mean().item())

        # 4. Add & Norm 2
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # print("Post Add & Norm 2 Mean:", src.mean().item())

        return src


def backward_hook(module, grad_input, grad_output):
    print("Grad input:", grad_input)
    print("Grad output:", grad_output)
    # You can perform additional logging or operations here


def tensor_backward_hook(grad):
    print("Grad:", grad)
    # Additional operations here


# Attaching the hook to a tensor

# def loss_function_risk_probabilities(risk, E):
#     if risk.shape[0] == 0:
#         NLL = torch.tensor(0., requires_grad=True)
#     else:
#         probability =torch.nn.Softmax()(risk)
#         all_prob= torch.cumsum(probability, 1)
#         uncensored_prob = all_prob * E.view(-1, 1)
#         cif = all_prob / (1-uncensored_prob)

#         # risk = torch.clamp(risk, max=80)
#         # hazard_ratio = torch.exp(risk)
#         # log_risk = torch.log(torch.cumsum(hazard_ratio, 0))
#         # uncensored_likelihood = torch.transpose(risk, 1, 1) - log_risk
#         # censored_likelihood = uncensored_likelihood * E.view(-1, 1)
#         num_observed_events = torch.sum(E)
#         if num_observed_events == 0:
#             #            NLL = -torch.sum(censored_likelihood)
#             loss = torch.tensor(0., requires_grad=True)
#         else:
#             loss = torch.sum(E.view(-1, 1)*torch.log(probability/(1-uncensored_prob))+(1-E.view(-1, 1))*torch.log(1-cif**E.view(-1, 1)))
#             # loss = -torch.sum(censored_likelihood) / num_observed_events

#     return loss

def loss_function_risk_reg(risk, E):
    if risk.shape[0] == 0:
        NLL = torch.tensor(0., requires_grad=True)
    else:
        risk = torch.clamp(risk, max=80)
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, 0))
        uncensored_likelihood = torch.transpose(risk, 1, 1) - log_risk
        censored_likelihood = uncensored_likelihood * E.view(-1, 1)
        num_observed_events = torch.sum(E)
        if num_observed_events == 0:
            #            NLL = -torch.sum(censored_likelihood)
            NLL = torch.tensor(0., requires_grad=True)
        else:
            NLL = -torch.sum(censored_likelihood) / num_observed_events

    return NLL


class BaseSurvVAE(nn.Module):
    def __init__(self, input_dim, risk_nn_conf, risk_type, id_covariate, start_time_covariate,
                 time_covariate, event_covariate, model_type, beta, sigma, regression_ratio, dropout=0.,
                 time_bins=[0, 0, 0], d_model=32, num_head=2, num_transformer_layer=2, dim_feedforward=512,
                 pred_time=None, eval_time=None):
        ## input_dim: dimension of the input data
        ## risk_nn_conf: configuration of the risk network
        ## risk_type: type of the risk network

        super(BaseSurvVAE, self).__init__()
        print("Initializing BaseSurvVAE")
        self.risk_type = risk_type
        self.model_type = model_type
        self.time_covariate = time_covariate
        self.start_time_covariate = start_time_covariate
        self.event_covariate = event_covariate
        self.id_covariate = id_covariate
        self.beta = beta
        self.sigma = sigma
        self.time_min = time_bins[0]
        self.time_max = time_bins[1]
        self.time_bin_num = time_bins[2]
        self.num_Category = time_bins[2]
        self.regression_ratio = regression_ratio
        self.last_N = 0 if self.risk_type is None else int(self.risk_type.split('_')[-1])
        self.num_head = num_head
        self.num_transformer_layer = num_transformer_layer
        self.dim_feedforward = dim_feedforward
        self.pred_time = pred_time
        self.eval_time = eval_time
        self.d_model = d_model

        self.coxfitter = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)

        # self.D = layer_conf[0]
        self.D = input_dim
        # transformer = 1
        if 'transformer' in model_type:
            self.transformer = True
        else:
            self.transformer = False
        if self.transformer:
            # d_model = 32
            if 'long' in model_type:
                # layer_conf = [d_model+input_dim] + risk_nn_conf
                layer_conf = [d_model * self.num_Category] + risk_nn_conf
            else:
                layer_conf = [d_model] + risk_nn_conf
        else:
            layer_conf = [input_dim * self.last_N] + risk_nn_conf

        activations = {
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU
        }

        if self.transformer:
            self.feature_projector = Linear(input_dim, d_model)
            # encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=2,
            #                                  dim_feedforward=d_model)
            encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=num_head,
                                                     dim_feedforward=dim_feedforward, dropout=dropout)

            # # Build the transformer encoder
            #
            self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers,
                                                          num_layers=num_transformer_layer)

            # self.monitored_transformer_encoder = build_monitored_transformer_encoder(1, d_model, 2, d_model,
            #                                                                     dropout)

            # self.monitored_transformer_encoder = transformer_encoder(1, d_model, 2, d_model,
            #                                                                     dropout)

            # Initialize the attention layer
            if 'long' not in model_type:
                self.attention_layer = AttentionLayer(
                    feature_dim=d_model)  # d_model is the dimension of the encoder outputs

        self.hazard_net = nn.ModuleList()
        if len(layer_conf) == 1:
            self.hazard_net = nn.Sequential(*self.hazard_net)
            return
        i = 0
        for i in range(len(layer_conf) - 2):
            self.hazard_net.append(nn.Linear(layer_conf[i], layer_conf[i + 1], bias=False))
            self.hazard_net.append(nn.BatchNorm1d(layer_conf[i + 1]))
            self.hazard_net.append(nn.Dropout(dropout))
            self.hazard_net.append(activations['relu']())

        try:
            self.hazard_net.append(nn.Linear(layer_conf[i + 1], layer_conf[i + 2], bias=False))
        except:
            pass
        if len(self.hazard_net) == 0:
            self.hazard_net.append(nn.Linear(layer_conf[0], layer_conf[1], bias=False))
        self.hazard_net = nn.Sequential(*self.hazard_net)

        # hook = self.hazard_net.register_backward_hook(backward_hook)

    def risk_forward(self, structured_mu, src_mask, last_meas=None):
        # Shared functionality for both ConvVAE and SimpleVAE
        # print("This is a shared method.")

        # Project the features to match d_model
        src = self.feature_projector(structured_mu)
        max_seq_length = 20

        output = self.transformer_encoder(src, src_key_padding_mask=src_mask.bool().T)

        if 'long' not in self.model_type:
            output = self.attention_layer(output, src_mask.bool())
        else:
            output = output.transpose(0, 1).reshape(-1, self.num_Category * self.d_model)

        # if last_meas not in [None, []]:
        #     output = torch.cat((output, last_meas), dim=1)

        return self.hazard_net(output)

    def loss_Log_Likelihood(self, k, fc_mask1, fc_mask2, out, epsilon=1e-7):
        sigma3 = 1.0
        I_1 = torch.sign(k)
        # Sum across dimension 2 then dimension 1, keep dimension for broadcasting
        denom = 1 - torch.sum(torch.sum(fc_mask1 * out, dim=2), dim=1, keepdim=True)
        denom = torch.clamp(denom, min=1e-7, max=1. - 1e-7)

        tmp1 = torch.sum(torch.sum(fc_mask2 * out, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1 / denom)

        tmp2 = torch.sum(torch.sum(fc_mask2 * out, dim=2), dim=1, keepdim=True)
        tmp2 = (1. - I_1) * torch.log(tmp2 / denom)

        LOSS_1 = - torch.mean(tmp1 + sigma3 * tmp2)
        return LOSS_1

    def risk_loss_transformer(self, structured_mu, E, time_data, last_meas_time, last_maes, src_mask):
        # Shared functionality for both ConvVAE and SimpleVAE
        # print("This is a shared method.")

        # # Project the features to match d_model
        # risk = self.risk_forward(structured_mu,src_mask.bool())
        risk = self.risk_forward(structured_mu, src_mask.bool(), last_maes)

        sort_values = time_data * 10 ** 4 + E

        # Get the sorted indices
        sorted_indices = torch.argsort(sort_values, descending=True)

        # Using these indices to rearrange the risk and events tensors
        sorted_risk = risk[sorted_indices]
        sorted_E = E[sorted_indices]

        # Now you can use the sorted risk and events tensors in your loss function
        surv_nll = loss_function_risk_reg(sorted_risk, sorted_E)

        return surv_nll, risk

    def risk_loss_transformer_long(self, structured_mu, E, time_data, last_meas_time, last_maes, src_mask):
        # Shared functionality for both ConvVAE and SimpleVAE
        # print("This is a shared method.")

        # # Project the features to match d_model
        # risk = self.risk_forward(structured_mu,src_mask.bool(), last_maes)
        risk = self.risk_forward(structured_mu, src_mask.bool())

        out = torch.nn.Softmax(dim=1)(risk)

        fc_mask1 = f_get_fc_mask1(last_meas_time[:, None], 1, self.num_Category).to(out.device)
        fc_mask2 = f_get_fc_mask2(last_meas_time[:, None] + time_data[:, None], E[:, None], 1, self.num_Category).to(
            out.device)

        surv_nll = self.loss_Log_Likelihood(E, fc_mask1, fc_mask2, out)
        total_loss = surv_nll  #+ self.beta * torch.sum(self.loss_rank_2(time_data, out, E, last_meas_time))

        return total_loss, out

    def risk_loss_long(self, mu, E, time_data, last_meas_time, train_x, surv_covs):
        # Shared functionality for both ConvVAE and SimpleVAE
        # print("This is a shared method.")

        # # Project the features to match d_model
        # risk = self.risk_forward(structured_mu,src_mask.bool(), last_maes)
        survival_indexes = cox_loss_last_n(self.last_N, train_x, self.id_covariate,
                                           self.time_covariate, self.event_covariate)
        # Convert the series of lists into a flat numpy array of indices
        resulting_rows = []

        # Loop through each list of indices in survival_indexes
        for index_list in survival_indexes:
            # Retrieve the rows from mu tensor corresponding to the indices in index_list
            rows = mu[index_list]
            cov = surv_covs[index_list]
            rows = torch.cat((rows, cov), dim=1)
            # Concatenate these rows to form a new row
            new_row = torch.cat(tuple(rows), dim=0)
            resulting_rows.append(new_row)

        # Stack all the new rows to form the resulting tensor
        resulting_tensor = torch.stack(resulting_rows, dim=0)

        risk = self.hazard_net(resulting_tensor)

        out = torch.nn.Softmax(dim=1)(risk)

        fc_mask1 = f_get_fc_mask1(last_meas_time[:, None], 1, self.num_Category)
        fc_mask2 = f_get_fc_mask2(last_meas_time[:, None] + time_data[:, None], E[:, None], 1, self.num_Category)

        surv_nll = self.loss_Log_Likelihood(E, fc_mask1, fc_mask2, out)
        total_loss = surv_nll  #+ self.beta * torch.sum(self.loss_rank_2(time_data, out, E, last_meas_time))

        return total_loss, out

    def risk_loss(self, mu, train_x, surv_covs):
        # Shared functionality for both ConvVAE and SimpleVAE
        # print("This is a shared method.")
        if 'cox_last' in self.risk_type:
            # Cox loss
            survival_indexes = cox_loss_last_n(self.last_N, train_x, self.id_covariate,
                                               self.time_covariate, self.event_covariate)
            # Convert the series of lists into a flat numpy array of indices
            resulting_rows = []

            # Loop through each list of indices in survival_indexes
            for index_list in survival_indexes:
                # Retrieve the rows from mu tensor corresponding to the indices in index_list
                rows = mu[index_list]
                cov = surv_covs[index_list]
                rows = torch.cat((rows, cov), dim=1)
                # Concatenate these rows to form a new row
                new_row = torch.cat(tuple(rows), dim=0)
                resulting_rows.append(new_row)

            # Stack all the new rows to form the resulting tensor
            resulting_tensor = torch.stack(resulting_rows, dim=0)
            # Index the mu tensor using these indices
            # selected_mu_values = mu[indices]
            # Extracting risk and last indexes
            risk = self.hazard_net(resulting_tensor)
            last_indexes = [index_list[-1] for index_list in survival_indexes]

            # Retrieving the events and time data
            E = train_x[last_indexes, self.event_covariate]
            time_data = train_x[last_indexes, self.time_covariate] - train_x[last_indexes, self.start_time_covariate]

            # Getting the indices that would sort the time tensor
            # sorted_indices = torch.argsort(time_data, descending=True)
            # Calculate the sorting values
            sort_values = time_data * 10 ** 4 + E

            # Get the sorted indices
            sorted_indices = torch.argsort(sort_values, descending=True)

            # Using these indices to rearrange the risk and events tensors
            sorted_risk = risk[sorted_indices]
            sorted_E = E[sorted_indices]

            # Now you can use the sorted risk and events tensors in your loss function
            surv_nll = loss_function_risk_reg(sorted_risk, sorted_E)

            return surv_nll
        else:
            return 0

    def loss_rank(self, time_data, risk):
        N = time_data.shape[0]
        # sigma = 0.1

        # Expand time_data to create a matrix comparing every i with every j
        time_i = time_data.unsqueeze(1).expand(-1, N)
        time_j = time_data.unsqueeze(0).expand(N, -1)

        # Create the A matrix where A[i, j] is 1 if time_data[i] < time_data[j], using broadcasting
        A = (time_i < time_j).float()

        # Calculate the risk differences and apply the exponential function
        risk_diff = risk.reshape(-1).unsqueeze(1) - risk.reshape(-1).unsqueeze(0)
        risk_factor = torch.exp(-risk_diff / self.sigma)

        # Multiply by the A matrix to consider only cases where time_data[i] < time_data[j]
        risk_factor = A * risk_factor

        return risk_factor

    def loss_rank_2(self, time_data, out, E, meas_time):
        sigma1 = 0.1

        eta = []
        one_vector = torch.ones_like(time_data)
        # for e in range(self.num_Event):
        e = 0
        num_Event = 1
        I_2 = (E == (e + 1)).double()
        I_2 = torch.diag(I_2.squeeze())

        tmp_e = torch.nn.Parameter(out.clone().detach(), requires_grad=True)  # Slicing and reshaping

        fc_mask3 = f_get_fc_mask3(time_data[:, None] + meas_time[:, None], meas_time[:, None], self.num_Category)
        R = torch.matmul(tmp_e, torch.tensor(fc_mask3.T))  # Matrix multiplication

        diag_R = torch.diag(R)
        R = torch.outer(one_vector, diag_R) - R  # Outer subtraction

        T = torch.relu(torch.sign(torch.outer(one_vector, time_data) - torch.outer(time_data, one_vector)))

        T = torch.matmul(I_2, T)  # Mask with event occurrence

        tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)

        eta.append(tmp_eta)

        eta = torch.cat(eta, dim=1)  # Stack along the second dimension (events)
        eta = torch.mean(eta.view(-1, num_Event), dim=1, keepdim=True)

        risk_factor = torch.sum(eta)  # Sum over num_Events

        return risk_factor

    # def get_survival_data_deneme(self, regression_input, train_x, surv_covs,pred_time):
    #     # Cox loss
    #     survival_indexes = cox_loss_last_n(self.last_N, train_x, self.id_covariate,
    #                             self.time_covariate, self.event_covariate)
    #     # Convert the series of lists into a flat numpy array of indices
    #
    #     # Extracting risk and last indexes
    #     last_indexes = [index_list[-1] for index_list in survival_indexes]
    #
    #     resulting_rows = []
    #
    #     for index_list in survival_indexes:
    #         # Retrieve the rows from mu tensor corresponding to the indices in index_list
    #         rows = regression_input[index_list]
    #         cov = surv_covs[index_list]
    #         rows = torch.cat((rows, cov), dim=1)
    #         # Concatenate these rows to form a new row
    #         new_row = torch.cat(tuple(rows), dim=0)
    #         resulting_rows.append(new_row)
    #     #
    #     # Stack all the new rows to form the resulting tensor
    #     resulting_tensor = torch.stack(resulting_rows, dim=0)
    #
    #     # Retrieving the events and time data
    #     E = train_x[last_indexes, self.event_covariate]
    #     time_data = train_x[last_indexes, self.time_covariate] - train_x[last_indexes, self.start_time_covariate]
    #     surv_ids = train_x[last_indexes, self.id_covariate]
    #     last_measurement_times = train_x[last_indexes, self.start_time_covariate]
    #
    #     P = E.shape[0]
    #     L = self.time_bin_num
    #     D = self.D
    #     # structured_mu = torch.zeros(P, L, D)
    #     # mask_tensor = torch.zeros(P, L)
    #
    #     # Assuming min_time, max_time, and num_bins are given
    #     min_time = self.time_min
    #     max_time = self.time_max
    #     num_bins = self.time_bin_num  # The number of bins you want
    #
    #     # Calculate the bin size based on the min and max times
    #     bin_size = (max_time - min_time) / num_bins
    #
    #     # Initialize the structured data tensor and mask tensor
    #     structured_mu = torch.zeros(num_bins, P, D)
    #     padding_mask_tensor = torch.ones(num_bins, P, dtype=torch.bool)
    #
    #     for ind, subject_id in enumerate(surv_ids):
    #         # Get the rows for this subject
    #         indices = (train_x[:,self.id_covariate] == subject_id).nonzero(as_tuple=True)[0]
    #         times = train_x[:,self.start_time_covariate][indices]
    #         data = regression_input[indices]
    #         survcovs = surv_covs[indices]
    #
    #         # Concatenate data and covariates
    #         data_with_covs = torch.cat((data, survcovs), dim=1)
    #
    #         # Map times to bin indices
    #         bin_indices = ((times - min_time) / bin_size).long()  # This will give you the bin index for each time
    #         bin_indices[
    #             bin_indices >= num_bins] = num_bins - 1  # Ensure the bin index does not exceed the number of bins
    #
    #         # Aggregate the data into bins
    #         for bin_index in bin_indices.unique():
    #             if bin_index >= pred_time:
    #                 continue
    #             relevant_data = data_with_covs[bin_indices == bin_index]
    #
    #             # Take the mean, sum, or any other aggregation method for data in the same bin
    #             # Here we take the mean; you can replace it with your preferred aggregation method
    #             structured_mu[bin_index, ind] = relevant_data.mean(dim=0)
    #
    #             # Set the mask to indicate data presence in this bin
    #             padding_mask_tensor[bin_index, ind] = 0
    #
    #     # print(f"regression_input.device {regression_input.device}")
    #     # return (resulting_tensor, structured_mu.transpose(0,1).double().to(regression_input.device), E, time_data,
    #     #         padding_mask_tensor.transpose(0,1).to(regression_input.device), last_measurement_times)
    #     return (resulting_tensor, structured_mu.double().to(regression_input.device), E, time_data,
    #             padding_mask_tensor.to(regression_input.device), last_measurement_times)

    # def get_survival_data_deneme2(self, regression_input, train_x, surv_covs,pred_time):
    #     # Cox loss
    #     survival_indexes = cox_loss_last_n(self.last_N, train_x, self.id_covariate,
    #                             self.time_covariate, self.event_covariate)
    #     # Convert the series of lists into a flat numpy array of indices
    #
    #     # Extracting risk and last indexes
    #     last_indexes = [index_list[-1] for index_list in survival_indexes]
    #     events = train_x[last_indexes, self.event_covariate]
    #     last_stop_times = train_x[last_indexes, self.time_covariate]
    #
    #     mask = last_stop_times >= pred_time
    #     last_indexes = [index for index, keep in zip(last_indexes, mask) if keep]
    #     kept_ids = train_x[last_indexes, self.id_covariate]
    #
    #     resulting_rows = []
    #
    #     for i, index_list in enumerate(survival_indexes):
    #         if mask[i] == 0:
    #             continue
    #         # Retrieve the rows from mu tensor corresponding to the indices in index_list
    #         rows = regression_input[index_list]
    #         cov = surv_covs[index_list]
    #         rows = torch.cat((rows, cov), dim=1)
    #         # Concatenate these rows to form a new row
    #         new_row = torch.cat(tuple(rows), dim=0)
    #         resulting_rows.append(new_row)
    #     #
    #     # Stack all the new rows to form the resulting tensor
    #     resulting_tensor = torch.stack(resulting_rows, dim=0)
    #
    #
    #     # Retrieving the events and time data
    #     E = train_x[last_indexes, self.event_covariate]
    #     time_data = train_x[last_indexes, self.time_covariate] - train_x[last_indexes, self.start_time_covariate]
    #     surv_ids = train_x[last_indexes, self.id_covariate]
    #     last_measurement_times = train_x[last_indexes, self.start_time_covariate]
    #
    #     mask2 = torch.isin(train_x[:, self.id_covariate], kept_ids)
    #
    #     # Use the mask to filter train_x_all
    #     train_x = train_x[mask2]
    #     regression_input = regression_input[mask2, :]
    #
    #     P = E.shape[0]
    #     L = self.time_bin_num
    #     D = self.D
    #     # structured_mu = torch.zeros(P, L, D)
    #     # mask_tensor = torch.zeros(P, L)
    #
    #     # Assuming min_time, max_time, and num_bins are given
    #     min_time = self.time_min
    #     max_time = self.time_max
    #     num_bins = self.time_bin_num  # The number of bins you want
    #
    #     # Calculate the bin size based on the min and max times
    #     bin_size = (max_time - min_time) / num_bins
    #
    #     # Initialize the structured data tensor and mask tensor
    #     structured_mu = torch.zeros(num_bins, P, D)
    #     padding_mask_tensor = torch.ones(num_bins, P, dtype=torch.bool)
    #
    #     for ind, subject_id in enumerate(surv_ids):
    #         # Get the rows for this subject
    #         indices = (train_x[:,self.id_covariate] == subject_id).nonzero(as_tuple=True)[0]
    #         times = train_x[:,self.start_time_covariate][indices]
    #         data = regression_input[indices]
    #         survcovs = surv_covs[indices]
    #
    #         # Concatenate data and covariates
    #         data_with_covs = torch.cat((data, survcovs), dim=1)
    #
    #         # Map times to bin indices
    #         bin_indices = ((times - min_time) / bin_size).long()  # This will give you the bin index for each time
    #         bin_indices[
    #             bin_indices >= num_bins] = num_bins - 1  # Ensure the bin index does not exceed the number of bins
    #
    #         # Aggregate the data into bins
    #         for bin_index in bin_indices.unique():
    #             if bin_index >= pred_time:
    #                 continue
    #             relevant_data = data_with_covs[bin_indices == bin_index]
    #
    #             # Take the mean, sum, or any other aggregation method for data in the same bin
    #             # Here we take the mean; you can replace it with your preferred aggregation method
    #             structured_mu[bin_index, ind] = relevant_data.mean(dim=0)
    #
    #             # Set the mask to indicate data presence in this bin
    #             padding_mask_tensor[bin_index, ind] = 0
    #
    #     # print(f"regression_input.device {regression_input.device}")
    #     # return (resulting_tensor, structured_mu.transpose(0,1).double().to(regression_input.device), E, time_data,
    #     #         padding_mask_tensor.transpose(0,1).to(regression_input.device), last_measurement_times)
    #     return (resulting_tensor, structured_mu.double().to(regression_input.device), E, time_data,
    #             padding_mask_tensor.to(regression_input.device), last_measurement_times)

    def get_survival_data(self, regression_input, train_x, surv_covs):
        # Cox loss
        survival_indexes = cox_loss_last_n(self.last_N, train_x, self.id_covariate,
                                           self.time_covariate, self.event_covariate)
        # Convert the series of lists into a flat numpy array of indices

        # Extracting risk and last indexes
        last_indexes = [index_list[-1] for index_list in survival_indexes]

        resulting_rows = []

        for index_list in survival_indexes:
            # Retrieve the rows from mu tensor corresponding to the indices in index_list
            rows = regression_input[index_list]
            cov = surv_covs[index_list]
            rows = torch.cat((rows, cov), dim=1)
            # Concatenate these rows to form a new row
            new_row = torch.cat(tuple(rows), dim=0)
            resulting_rows.append(new_row)

        # Stack all the new rows to form the resulting tensor
        resulting_tensor = torch.stack(resulting_rows, dim=0)

        # Retrieving the events and time data
        E = train_x[last_indexes, self.event_covariate]
        time_data = train_x[last_indexes, self.time_covariate] - train_x[last_indexes, self.start_time_covariate]
        surv_ids = train_x[last_indexes, self.id_covariate]
        last_measurement_times = train_x[last_indexes, self.start_time_covariate]

        if self.transformer:
            P = E.shape[0]
            L = self.time_bin_num
            D = self.D
            # structured_mu = torch.zeros(P, L, D)
            # mask_tensor = torch.zeros(P, L)

            # Assuming min_time, max_time, and num_bins are given
            min_time = self.time_min
            max_time = self.time_max
            num_bins = self.time_bin_num  # The number of bins you want

            # Calculate the bin size based on the min and max times
            bin_size = (max_time - min_time) / num_bins

            # Initialize the structured data tensor and mask tensor
            structured_mu = torch.zeros(num_bins, P, D)
            padding_mask_tensor = torch.ones(num_bins, P, dtype=torch.bool)

            for ind, subject_id in enumerate(surv_ids):
                # Get the rows for this subject
                indices = (train_x[:, self.id_covariate] == subject_id).nonzero(as_tuple=True)[0]
                times = train_x[:, self.start_time_covariate][indices]
                data = regression_input[indices]
                survcovs = surv_covs[indices]

                # Concatenate data and covariates
                data_with_covs = torch.cat((data, survcovs), dim=1)

                # Map times to bin indices
                bin_indices = ((times - min_time) / bin_size).long()  # This will give you the bin index for each time
                bin_indices[
                    bin_indices >= num_bins] = num_bins - 1  # Ensure the bin index does not exceed the number of bins

                # Aggregate the data into bins
                for bin_index in bin_indices.unique():
                    relevant_data = data_with_covs[bin_indices == bin_index]

                    # Take the mean, sum, or any other aggregation method for data in the same bin
                    # Here we take the mean; you can replace it with your preferred aggregation method
                    structured_mu[bin_index, ind] = relevant_data.mean(dim=0)

                    # Set the mask to indicate data presence in this bin
                    padding_mask_tensor[bin_index, ind] = 0
            structured_mu = structured_mu.double().to(regression_input.device)
            padding_mask_tensor = padding_mask_tensor.to(regression_input.device)
        else:
            structured_mu = None
            padding_mask_tensor = None

        # print(f"regression_input.device {regression_input.device}")
        # return (resulting_tensor, structured_mu.transpose(0,1).double().to(regression_input.device), E, time_data,
        #         padding_mask_tensor.transpose(0,1).to(regression_input.device), last_measurement_times)
        return (resulting_tensor, structured_mu, E, time_data,
                padding_mask_tensor, last_measurement_times)


#     def get_survival_data_pred(self, regression_input, train_x_all_orj, surv_covs, pred_time):
#
#         ## pred_times: prediction times for the survival data
#         ## Here we update the input data according to pred_times. We cut the data at the prediction times
#         train_x_all = train_x_all_orj.clone()
#
#         # pred_time = 38
#
# #         mask_2 = (train_x_all[:,self.time_covariate] < pred_time) #& (train_x_all[:,self.event_covariate] == 1)
# #         exclude_ids = train_x_all[mask_2][:, self.id_covariate]
# #
# # ### Exclude the ids that have event before the pred_time
# #         mask = torch.ones(train_x_all.shape[0], dtype=bool)
# #         for id in exclude_ids:
# #             mask = mask & (train_x_all[:, self.id_covariate] != id)
# #         train_x_all = train_x_all[mask,:]
# #         regression_input_pred = regression_input[mask,:]
# # #         regression_input_pred = regression_input.clone()
#
#         survival_indexes_all = cox_loss_last_n(self.last_N, train_x_all, self.id_covariate,
#                                 self.time_covariate, self.event_covariate)
#         last_stop_times = train_x_all[survival_indexes_all, self.time_covariate]
#         last_obs_times = train_x_all[survival_indexes_all, self.start_time_covariate]
#         # last_stop_ages= train_x_all[survival_indexes_all, 3]
#         #
#         # mask = last_stop_ages <= pred_time
#         mask = last_obs_times >= pred_time
#         included_ids = train_x_all[survival_indexes_all][~mask[:,0], 0, :]
#         # Extract the id_covariate column
#         id_covariate_column = train_x_all[:, self.id_covariate]
#
#         # Create a mask that checks if values in id_covariate_column are in included_ids
#         mask = torch.isin(id_covariate_column, included_ids[:,self.id_covariate ])
#
#         # Use the mask to filter train_x_all
#         train_x_all = train_x_all[mask]
#         regression_input_pred = regression_input[mask, :]
#
#         ids = train_x_all[:, self.id_covariate]
#         events = train_x_all[:, self.event_covariate]
#
#         # Prepare a tensor to hold the updated event information
#         updated_events = torch.zeros_like(events)
#
#         # Update the event information for each ID
#         unique_ids = ids.unique()
#         for uid in unique_ids:
#             # Find the last index of the current ID
#             last_idx = (ids == uid).nonzero(as_tuple=True)[0][-1]
#             # Update all rows with the event from the last row of the current ID
#             updated_events[ids == uid] = events[last_idx]
#
#         # Replace the event information in the original data
#         train_x_all[:, self.event_covariate] = updated_events
#         # pred_time = 30
#         # mask = train_x_all[:,self.start_time_covariate] <= pred_time
#         #
#         # train_x = train_x_all[mask,:]
#         # regression_input_pred = regression_input_pred[mask,:]
#
#         train_x = train_x_all.clone()
#
#         # Cox loss
#         survival_indexes = cox_loss_last_n(self.last_N, train_x, self.id_covariate,
#                                 self.time_covariate, self.event_covariate)
#         # Convert the series of lists into a flat numpy array of indices
#
#         # Extracting risk and last indexes
#         last_indexes = [index_list[-1] for index_list in survival_indexes]
#
#         resulting_rows = []
#
#         for ind, index_list in enumerate(survival_indexes):
#             # Retrieve the rows from mu tensor corresponding to the indices in index_list
#             rows = regression_input_pred[index_list]
#             train_x[index_list, self.time_covariate] = last_stop_times[ind]
#             cov = surv_covs[index_list]
#             rows = torch.cat((rows, cov), dim=1)
#             # Concatenate these rows to form a new row
#             new_row = torch.cat(tuple(rows), dim=0)
#             resulting_rows.append(new_row)
#
#         # Stack all the new rows to form the resulting tensor
#         resulting_tensor = torch.stack(resulting_rows, dim=0)
#
#         # Retrieving the events and time data
#         E = train_x[last_indexes, self.event_covariate]
#         time_data = train_x[last_indexes, self.time_covariate] - train_x[last_indexes, self.start_time_covariate]
#         surv_ids = train_x[last_indexes, self.id_covariate]
#         last_measurement_times = train_x[last_indexes, self.start_time_covariate]
#
#         P = E.shape[0]
#         L = self.time_bin_num
#         D = self.D
#         # structured_mu = torch.zeros(P, L, D)
#         # mask_tensor = torch.zeros(P, L)
#
#         # Assuming min_time, max_time, and num_bins are given
#         min_time = self.time_min
#         max_time = self.time_max
#         num_bins = self.time_bin_num  # The number of bins you want
#
#         # Calculate the bin size based on the min and max times
#         bin_size = (max_time - min_time) / num_bins
#
#         # Initialize the structured data tensor and mask tensor
#         structured_mu = torch.zeros(num_bins, P, D)
#         padding_mask_tensor = torch.ones(num_bins, P, dtype=torch.bool)
#
#         for ind, subject_id in enumerate(surv_ids):
#             # Get the rows for this subject
#             indices = (train_x[:,self.id_covariate] == subject_id).nonzero(as_tuple=True)[0]
#             times = train_x[:,self.start_time_covariate][indices]
#             data = regression_input_pred[indices]
#             survcovs = surv_covs[indices]
#
#             # Concatenate data and covariates
#             data_with_covs = torch.cat((data, survcovs), dim=1)
#
#             # Map times to bin indices
#             bin_indices = ((times - min_time) / bin_size).long()  # This will give you the bin index for each time
#             bin_indices[
#                 bin_indices >= num_bins] = num_bins - 1  # Ensure the bin index does not exceed the number of bins
#
#             # Aggregate the data into bins
#             for bin_index in bin_indices.unique():
#                 relevant_data = data_with_covs[bin_indices == bin_index]
#
#                 # Take the mean, sum, or any other aggregation method for data in the same bin
#                 # Here we take the mean; you can replace it with your preferred aggregation method
#                 structured_mu[bin_index, ind] = relevant_data.mean(dim=0)
#
#                 # Set the mask to indicate data presence in this bin
#                 padding_mask_tensor[bin_index, ind] = 0
#
#         # print(f"regression_input.device {regression_input.device}")
#         # return (resulting_tensor, structured_mu.transpose(0,1).double().to(regression_input.device), E, time_data,
#         #         padding_mask_tensor.transpose(0,1).to(regression_input.device), last_measurement_times)
#         return (resulting_tensor, structured_mu.double().to(regression_input.device), E, time_data,
#                 padding_mask_tensor.to(regression_input.device), last_measurement_times)


class ConvVAE(BaseSurvVAE):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, id_covariate, start_time_covariate, time_covariate, event_covariate,
                 model_type, beta, sigma, regression_ratio, surv_x_dim=0, vy_init=1, vy_fixed=False,
                 p_input=0., p=0.5, risk_type='cox_last_1', risk_nn_conf=[], time_bins=[0, 0, 0], d_model=32,
                 num_head=2, num_transformer_layer=2, dim_feedforward=512, pred_time=None, eval_time=None):
        super(ConvVAE, self).__init__(latent_dim + surv_x_dim, risk_nn_conf, risk_type, id_covariate,
                                      start_time_covariate, time_covariate,
                                      event_covariate, model_type, beta, sigma, regression_ratio, p, time_bins,
                                      d_model, num_head, num_transformer_layer, dim_feedforward, pred_time, eval_time)

        self.latent_dim = latent_dim
        self.input_dim = num_dim
        self.p_input = p_input
        self.p = p

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_1 = nn.Dropout2d(p=self.p)  # spatial dropout

        # second convolution layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_2 = nn.Dropout2d(p=self.p)

        self.fc1 = nn.Linear(32 * 9 * 9, 300)
        self.dropout1 = nn.Dropout(p=self.p)
        self.fc21 = nn.Linear(300, 30)
        self.dropout2 = nn.Dropout(p=self.p)
        self.fc211 = nn.Linear(30, self.latent_dim)
        self.fc221 = nn.Linear(30, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(latent_dim, 30)
        self.dropout3 = nn.Dropout(p=self.p)
        self.fc31 = nn.Linear(30, 300)
        self.dropout4 = nn.Dropout(p=self.p)
        self.fc4 = nn.Linear(300, 32 * 9 * 9)

        self.dropout2d_3 = nn.Dropout2d(p=self.p)
        # first transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        self.dropout2d_4 = nn.Dropout2d(p=self.p)
        # second transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        # convolution
        z = F.relu(self.conv1(x))
        z = self.dropout2d_1(self.pool1(z))
        z = F.relu(self.conv2(z))
        z = self.dropout2d_2(self.pool2(z))

        # MLP
        z = z.view(-1, 32 * 9 * 9)
        h1 = self.dropout1(F.relu(self.fc1(z)))
        h2 = self.dropout2(F.relu(self.fc21(h1)))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        # MLP
        x = self.dropout3(F.relu(self.fc3(z)))
        x = self.dropout4(F.relu(self.fc31(x)))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = self.dropout2d_3(x.view(-1, 32, 9, 9))
        x = self.dropout2d_4(F.relu(self.deconv1(x)))
        return torch.sigmoid(self.deconv2(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample_latent(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.input_dim), x.view(-1, self.input_dim)),
                       mask.view(-1, self.input_dim))
        mask_sum = torch.sum(mask.view(-1, self.input_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        # nll = se / (2 * torch.exp(self._log_vy))
        # nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        nll = se / (2 * torch.exp(log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + log_vy)
        return mse, torch.sum(nll, dim=1)


class SimpleVAE(BaseSurvVAE):
    """
    Encoder and decoder for variational autoencoder with simple multi-layered perceptrons.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, input_dim, id_covariate, start_time_covariate, time_covariate, event_covariate,
                 model_type
                 , beta, sigma, regression_ratio, surv_x_dim=0, vy_init=1, vy_fixed=False,
                 risk_type='cox_last_1', risk_nn_conf=[], vae_nn_conf=[], dropout=0., time_bins=[0, 0, 0],
                 d_model=32, num_head=2, num_transformer_layer=2, dim_feedforward=512, pred_time=None, eval_time=None):
        super(SimpleVAE, self).__init__(latent_dim + surv_x_dim, risk_nn_conf, risk_type, id_covariate,
                                        start_time_covariate, time_covariate,
                                        event_covariate, model_type, beta, sigma, regression_ratio, dropout,
                                        time_bins, d_model, num_head, num_transformer_layer, dim_feedforward, pred_time,
                                        eval_time)

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(input_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        if vae_nn_conf is None:
            vae_nn_conf = [None, None]
        h_dim_e = vae_nn_conf[0]
        h_dim_d = vae_nn_conf[1]

        e_lin_layers = nn.ModuleList()
        # e_lin_layers.append(torch.nn.Softmax())
        next_input_dim = self.input_dim
        if h_dim_e is not None and h_dim_e != [] and h_dim_e != 0 and h_dim_e != [0]:
            neurons = [self.input_dim, *h_dim_e]
            for i in range(len(neurons) - 1):
                e_lin_layers.append(nn.Linear(neurons[i], neurons[i + 1]))
                torch.nn.init.normal_(e_lin_layers[-1].weight, mean=0.0, std=.05)
                torch.nn.init.normal_(e_lin_layers[-1].bias, mean=0.0, std=.05)
                e_lin_layers.append(nn.Dropout(dropout))
                e_lin_layers.append(nn.ReLU())
            next_input_dim = h_dim_e[-1]

        self.encoder_layers = nn.Sequential(*e_lin_layers)

        self.mean_layer = nn.ModuleList()
        self.mean_layer.append(nn.Linear(next_input_dim, latent_dim))
        torch.nn.init.normal_(self.mean_layer[0].weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.mean_layer[0].bias, mean=0.0, std=.05)
        self.mean_layer = nn.Sequential(*self.mean_layer)

        self.log_var_layer = nn.ModuleList()
        self.log_var_layer.append(nn.Linear(next_input_dim, latent_dim))
        torch.nn.init.normal_(self.log_var_layer[0].weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.log_var_layer[0].bias, mean=0.0, std=.05)
        self.log_var_layer = nn.Sequential(*self.log_var_layer)

        d_lin_layers = nn.ModuleList()
        next_input_dim = latent_dim
        if h_dim_d is not None and h_dim_d != [] and h_dim_d != 0 and h_dim_d != [0]:
            neurons = [latent_dim, *h_dim_d]
            for i in range(len(neurons) - 1):
                d_lin_layers.append(nn.Linear(neurons[i], neurons[i + 1]))
                torch.nn.init.normal_(d_lin_layers[-1].weight, mean=0.0, std=.05)
                torch.nn.init.normal_(d_lin_layers[-1].bias, mean=0.0, std=.05)
                d_lin_layers.append(nn.Dropout(dropout))
                d_lin_layers.append(nn.ReLU())
            next_input_dim = h_dim_d[-1]

        d_lin_layers.append(nn.Linear(next_input_dim, self.input_dim))
        # d_lin_layers.append(nn.Softmax())
        self.decoder_layers = nn.Sequential(*d_lin_layers)

        # # encoder network
        # self.fc1 = nn.Linear(num_dim, 300)
        # self.fc21 = nn.Linear(300, 30)
        # self.fc211 = nn.Linear(30, latent_dim)
        # self.fc221 = nn.Linear(30, latent_dim)
        #
        # # decoder network
        # self.fc3 = nn.Linear(latent_dim, 30)
        # self.fc31 = nn.Linear(30, 300)
        # self.fc4 = nn.Linear(300, num_dim)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        e_out = self.encoder_layers(x)
        return self.mean_layer(e_out), self.log_var_layer(e_out)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        return self.decoder_layers(z)

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.sample_latent(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.input_dim), x.view(-1, self.input_dim)),
                       mask.view(-1, self.input_dim))
        mask_sum = torch.sum(mask.view(-1, self.input_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)


if __name__ == "__main__":
    """
    This is used for pre-training.
    """

    subjects_per_batch = 20

    # create parser and set variables
    opt = VAEArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    assert loss_function == 'mse' or loss_function == 'nll', ("Unknown loss function " + loss_function)
    assert ('T' in locals() and T is not None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    folder_exists = os.path.isdir(save_path)
    if not folder_exists:
        os.makedirs(save_path)

    # set up dataset
    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':
            dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                             mask_file=mask_file, root_dir=data_source_path,
                                             transform=transforms.ToTensor(), x_order=x_order,
                                             surv_x=surv_x, cut_time=cut_time, id_covariate=id_covariate)
            test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data, csv_file_label=csv_file_test_label,
                                                  mask_file=test_mask_file, root_dir=data_source_path,
                                                  transform=transforms.ToTensor(), x_order=x_order,
                                                  surv_x=surv_x, cut_time=cut_time, id_covariate=id_covariate)
            if csv_file_validation_data:
                validation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_validation_data,
                                                            csv_file_label=csv_file_validation_label,
                                                            mask_file=validation_mask_file,
                                                            root_dir=data_source_path,
                                                            transform=transforms.ToTensor(), x_order=x_order,
                                                            surv_x=surv_x, cut_time=cut_time, id_covariate=id_covariate)
        elif dataset_type == 'RotatedMNIST':
            dataset = RotatedMNISTDatasetConv(data_file=csv_file_data,
                                              label_file=csv_file_label,
                                              mask_file=mask_file, root_dir=data_source_path,
                                              transform=transforms.ToTensor(), x_order=x_order,
                                              surv_x=surv_x, cut_time=cut_time)

    elif type_nnet == 'simple':
        if dataset_type == 'HealthMNIST':
            dataset = HealthMNISTDataset(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                         mask_file=mask_file, root_dir=data_source_path,
                                         transform=transforms.ToTensor(), x_order=x_order,
                                         surv_x=surv_x, cut_time=cut_time)
            test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data, csv_file_label=csv_file_test_label,
                                              mask_file=test_mask_file, root_dir=data_source_path,
                                              transform=transforms.ToTensor(), x_order=x_order,
                                              surv_x=surv_x, cut_time=cut_time)
            if csv_file_validation_data:
                validation_dataset = HealthMNISTDataset(csv_file_data=csv_file_validation_data,
                                                        csv_file_label=csv_file_validation_label,
                                                        mask_file=validation_mask_file,
                                                        root_dir=data_source_path,
                                                        transform=transforms.ToTensor(), x_order=x_order,
                                                        surv_x=surv_x, cut_time=cut_time)
        elif dataset_type == 'RotatedMNIST':
            dataset = RotatedMNISTDataset(data_file=csv_file_data,
                                          label_file=csv_file_label,
                                          mask_file=mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor(), x_order=x_order,
                                          surv_x=surv_x, cut_time=cut_time)
        elif dataset_type == 'HUSCorogene':
            dataset = HUSCorogeneDataset(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                         mask_file=mask_file, root_dir=data_source_path, x_order=x_order,
                                         surv_x=surv_x, cut_time=cut_time)
            mean_min = dataset.mean_min
            std_max = dataset.std_max
            test_dataset = HUSCorogeneDataset(csv_file_data=csv_file_test_data, csv_file_label=csv_file_test_label,
                                              mask_file=test_mask_file, root_dir=data_source_path,
                                              mean_min=mean_min, std_max=std_max, x_order=x_order,
                                              surv_x=surv_x, cut_time=cut_time)
            if csv_file_validation_data:
                validation_dataset = HUSCorogeneDataset(csv_file_data=csv_file_validation_data,
                                                        csv_file_label=csv_file_validation_label,
                                                        mask_file=validation_mask_file,
                                                        root_dir=data_source_path,
                                                        mean_min=mean_min, std_max=std_max, x_order=x_order,
                                                        surv_x=surv_x, cut_time=cut_time)

    print('Length of dataset:  {}'.format(len(dataset)))
    Q = len(dataset[0]['label'])

    # set up Data Loader
    # dataloader = DataLoader(dataset, min(len(dataset), 256), shuffle=True, num_workers=0)
    dataloader = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(
        VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=num_workers)

    vy = torch.Tensor(np.ones(num_dim) * vy_init)

    # set up model and send to GPU if available
    if type_nnet == 'conv':
        print('Using convolutional neural network')
        # nnet_model = ConvVAE(latent_dim, num_dim, id_covariate, start_time_covariate, time_covariate, event_covariate,
        #                      surv_x_dim=len(surv_x), model_type=model_type, vy_init=vy_init, vy_fixed=vy_fixed,
        #                      p_input=dropout_input, p=dropout, risk_type=risk_type,
        #                      risk_nn_conf=risk_nn_conf).double().to(device)
        nnet_model = ConvVAE(latent_dim, num_dim, id_covariate, start_time_covariate, time_covariate, event_covariate,
                             surv_x_dim=len(surv_x),
                             model_type=model_type, beta=beta, sigma=sigma, regression_ratio=regression_ratio,
                             vy_init=vy_init,
                             vy_fixed=vy_fixed,
                             p_input=dropout_input, p=dropout, risk_type=risk_type, risk_nn_conf=risk_nn_conf,
                             time_bins=time_bins,
                             d_model=d_model,
                             num_head=num_head, dim_feedforward=dim_feedforward,
                             num_transformer_layer=num_transformer_layer,
                             pred_time=pred_time, eval_time=eval_time).double().to(device)

    elif type_nnet == 'simple':
        print('Using standard MLP')
        # nnet_model = SimpleVAE(latent_dim, num_dim, vy, vy_fixed).to(device)
        nnet_model = SimpleVAE(latent_dim, num_dim, id_covariate, start_time_covariate, time_covariate,
                               event_covariate, beta=beta, sigma=sigma, regression_ratio=regression_ratio,
                               surv_x_dim=len(surv_x), model_type=model_type,
                               risk_nn_conf=risk_nn_conf, vae_nn_conf=[vae_nn_enc, vae_nn_dec],
                               vy_init=vy_init, vy_fixed=vy_fixed, dropout=dropout, time_bins=time_bins,
                               d_model=d_model,
                               num_head=num_head, dim_feedforward=dim_feedforward,
                               num_transformer_layer=num_transformer_layer,
                               pred_time=pred_time, eval_time=eval_time).double().to(device)

    # Load pre-trained encoder/decoder parameters if present
    try:
        nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))
        print('Loaded pre-trained values.')
    except:
        print('Did not load pre-trained values.')

    optimiser = torch.optim.Adam(nnet_model.parameters(), lr=1e-3)

    print(nnet_model)

    if epochs not in [0, 1, 2] and not is_early_stopping:
        pd.to_pickle(opt,
                     os.path.join(save_path, 'arguments.pkl'))

    # if nnet_model.model_type == 'VAE':
    #     regression_ratio = 0.
    # elif nnet_model.model_type == 'VAE_survival':
    #     regression_ratio = .999

    validation_interval = 5
    validation_results = []
    validation_cindex_results = []
    best_value = np.inf
    best_val_cindex = 0

    net_train_loss = np.empty((0, 1))
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # start training VAE
        nnet_model.train()
        train_loss = 0
        recon_loss_sum = 0
        nll_loss = 0
        kld_loss = 0
        survival_loss_sum = 0

        for batch_idx, sample_batched in enumerate(dataloader):
            data = sample_batched['digit'].double()
            data = data.to(device)  # send to GPU
            mask = sample_batched['mask'].double()
            mask = mask.to(device)
            label = sample_batched['label'].double().to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate + 1:]), dim=1)
            surv_covs = sample_batched['surv_covariates'].double().to(device)

            optimiser.zero_grad()  # clear gradients

            recon_batch, mu, log_var = nnet_model(data)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)

            if nnet_model.model_type == 'VAE':
                survival_loss = torch.tensor(0.0, dtype=torch.double).to(device)
            elif 'VAE_survival' in nnet_model.model_type:
                if nnet_model.transformer:
                    last_meas, structured_mu, E, time_data, src_padding_mask, last_meas_time = nnet_model.get_survival_data(
                        nnet_model.sample_latent(mu, log_var),
                        label, surv_covs)

                    survival_loss, out = nnet_model.risk_loss_transformer(structured_mu.double(), E, time_data,
                                                                          last_meas_time, last_meas,
                                                                          src_padding_mask.double())
                else:
                    ### Here the survival loss is the average of observed events
                    survival_loss = nnet_model.risk_loss(nnet_model.sample_latent(mu, log_var), label, surv_covs)
                # Assuming 'tensor' is the tensor you want to track

            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            if loss_function == 'nll':
                survival_loss_sum += survival_loss.item()
                # net_loss = (nll_loss + kld_loss) * (1 - regression_ratio) + regression_ratio * survival_loss
                # Calculate the two parts of the loss
                part1 = torch.mean((nll + KLD))
                part2 = survival_loss

                # if epoch > 200:
                #     balancing_ratio = torch.abs(part1 / (part1 + part2)).item()
                #     loss = part1 * (1 - regression_ratio) * (1-balancing_ratio) + regression_ratio * balancing_ratio * part2
                #     if not epoch % 10 and batch_idx == 0:
                #         print(f'UPDATED BALLANCE RATIO: {balancing_ratio}')
                # else:
                #     loss = part1 #* (1 - regression_ratio) + regression_ratio * part2

                # regression_ratio = .999
                loss = part1 * (1 - regression_ratio) + regression_ratio * part2

                # loss = torch.mean((nll + KLD)) * (1-regression_ratio) + regression_ratio * survival_loss
                # survival_loss_sum += survival_loss
            elif loss_function == 'mse':
                loss = torch.sum(recon_loss + KLD)

            if torch.isnan(loss).any():
                print("net_loss contains NaN values!")
                survival_loss = nnet_model.risk_loss(nnet_model.sample_latent(mu, log_var), label, surv_covs)

            loss.backward()  # compute gradients
            train_loss += loss.item()
            recon_loss_sum += recon_loss.sum().item()
            nll_loss += nll.sum().item()
            kld_loss += KLD.sum().item()

            optimiser.step()  # update parameters

        print(
            '====> Epoch: {}/{} - Duration: {:.4f}  - Average loss: {:.4f}  - KLD loss: {:.3f}  - NLL loss: {:.3f}  - Recon loss: {:.3f} - Survival Loss: {:.4f}'.format(
                epoch, epochs, time.time() - start_time, train_loss, kld_loss, nll_loss, recon_loss_sum,
                survival_loss_sum))
        net_train_loss = np.append(net_train_loss, train_loss)

        # if epoch > 1:
        #     # log metrics to wandb
        #     wandb.log({"net_loss_sum": train_loss,
        #                "recon_loss_sum": recon_loss_sum, "nll_loss_sum": nll_loss,
        #                "kld_loss_sum": kld_loss,
        #                "survival_loss_sum": survival_loss_sum })

        if (csv_file_validation_data and not (epoch % validation_interval)) or is_early_stopping:  # and epoch > 100:
            print("Testing the model with a validation set")
            batch_size_val = len(validation_dataset)
            validation_dataloader = HensmanDataLoader(validation_dataset, batch_sampler=VaryingLengthBatchSampler(
                VaryingLengthSubjectSampler(validation_dataset, nnet_model.id_covariate), subjects_per_batch),
                                                      num_workers=num_workers)
            validation_loss = torch.tensor(0.).to(device)
            validation_recon_loss = torch.tensor(0.).to(device)
            full_mu = torch.zeros(len(validation_dataset), nnet_model.latent_dim,
                                  dtype=next(nnet_model.parameters()).dtype).to(device)
            full_label = torch.zeros(len(validation_dataset), label.shape[1],
                                     dtype=next(nnet_model.parameters()).dtype).to(device)
            full_surv_covs = torch.zeros(len(validation_dataset), len(surv_x),
                                         dtype=next(nnet_model.parameters()).dtype).to(device)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    indices = sample_batched['idx']
                    data = sample_batched['digit'].double().to(device)
                    mask = sample_batched['mask'].to(device)
                    masked_data = torch.mul(data, mask.reshape(data.shape))
                    label = sample_batched['label'].double().to(device)
                    surv_covs = sample_batched['surv_covariates'].double().to(device)
                    recon_batch, mu_qz, log_var_qz = nnet_model(masked_data)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, masked_data, mask)
                    full_mu[indices] = mu_qz
                    full_label[indices] = label
                    full_surv_covs[indices] = surv_covs

                    KLD = -0.5 * torch.sum(1 + log_var_qz - mu_qz.pow(2) - log_var_qz.exp(), dim=1)
                    if nnet_model.model_type == 'VAE':
                        survival_loss = torch.tensor(0.0, dtype=torch.double).to(device)
                    elif nnet_model.model_type == 'VAE_survival':
                        survival_loss = nnet_model.risk_loss(mu_qz, label, surv_covs)

                    validation_loss += torch.mean((nll + KLD)) * (
                                1 - regression_ratio) + regression_ratio * survival_loss
                    validation_recon_loss += recon_loss.sum().item()
            print("Calculating C-index for validation set")
            try:
                c_index_val = calculate_risk_score(nnet_model, full_mu, full_label, full_surv_covs,
                                                   save_path=os.path.join(save_path, 'validation_results'))
            except:
                c_index_val = 0.
                print("Error calculating C-index for validation set")
            # validation_loss = torch.sum(total_loss)
            if validation_criteria == 'nll':
                validation_results.append(validation_loss)
            elif validation_criteria == 'c_index':
                validation_results.append(-c_index_val)
            elif validation_criteria == 'mse':
                validation_results.append(validation_recon_loss)

            # validation_cindex_results.append(validation_loss)
            if is_early_stopping:
                torch.save(full_mu, os.path.join(save_path, 'z_valid_best.pth'))
            else:
                torch.save(full_mu, os.path.join(save_path, 'z_valid.pth'))
            if is_early_stopping:
                pred_results = np.array(
                    [validation_loss.detach().cpu().numpy(), validation_recon_loss.detach().cpu().numpy(),
                     c_index_val])
                df_results = pd.DataFrame(pred_results[:, None], index=["NLL", "MSE", "C-Index"])

                # Save to CSV
                df_results.to_csv(os.path.join(save_path, "best_validation_scores_early_stopping.csv"), index=True,
                                  header=False)
            if epoch > 10:
                current_value = validation_results[-1]
                if current_value < best_value:
                    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'model_params_early_best.pth'),
                               _use_new_zipfile_serialization=False)
                    best_epoch = epoch
                    best_value = current_value
                    best_val_cindex = c_index_val
                    np.savetxt(os.path.join(save_path, 'best_epoch.csv'), np.array(epoch).reshape([1, 1]))
                    print('Best model saved at epoch {}'.format(epoch))
                    # Stack the results
                    pred_results = np.array(
                        [validation_loss.detach().cpu().numpy(), validation_recon_loss.detach().cpu().numpy(),
                         c_index_val])
                    df_results = pd.DataFrame(pred_results[:, None], index=["NLL", "MSE", "C-Index"])

                    # Save to CSV
                    df_results.to_csv(os.path.join(save_path, "best_validation_scores.csv"), index=True,
                                      header=False)

        if epoch % 50 == 0:
            print(nnet_model.vy)
            if run_tests:
                # VAEtest(test_dataset, nnet_model, type_nnet, id_covariate)
                VAEoutput(nnet_model, test_dataset, epoch, save_path, type_nnet, id_covariate, num_workers,
                          is_early_stopping=is_early_stopping)
            # torch.save(nnet_model.state_dict(), os.path.join(save_path, 'model_params_vae_' + str(epoch) + '.pth'))

    if run_tests:
        # VAEtest(test_dataset, nnet_model, type_nnet, id_covariate)
        VAEoutput(nnet_model, test_dataset, epoch, save_path, type_nnet, id_covariate, num_workers,
                  is_early_stopping=is_early_stopping)

    if is_early_stopping:
        print("Testing the model with a validation set")
        batch_size_val = len(validation_dataset)
        subjects_per_batch = validation_dataset.label_source['subject'].nunique()
        # validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_val, shuffle=False,
        #                                    num_workers=0)
        validation_dataloader = HensmanDataLoader(validation_dataset, batch_sampler=VaryingLengthBatchSampler(
            VaryingLengthSubjectSampler(validation_dataset, nnet_model.id_covariate), subjects_per_batch),
                                                  num_workers=num_workers)
        # validation_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=num_workers)
        total_loss = []
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(validation_dataloader):
                # indices = sample_batched['idx']
                data = sample_batched['digit'].to(device).double()
                mask = sample_batched['mask'].to(device).double()
                masked_data = torch.mul(data, mask.reshape(data.shape))
                label = sample_batched['label'].to(device).double()
                surv_covs = sample_batched['surv_covariates'].double().to(device)
                recon_batch, mu_qz, log_var_qz = nnet_model(masked_data)
                [recon_loss, nll] = nnet_model.loss_function(recon_batch, masked_data, mask)
                KLD = -0.5 * torch.sum(1 + log_var_qz - mu_qz.pow(2) - log_var_qz.exp(), dim=1)
                total_loss.append(nll + KLD)
        validation_loss = torch.sum(torch.cat(total_loss))
        validation_results.append(validation_loss)
        torch.save(mu_qz, os.path.join(save_path, 'z_valid_best.pth'))

        print("Calculating C-index for validation set")
        try:
            c_index_val = calculate_risk_score(nnet_model, mu_qz, label, surv_covs,
                                               save_path=os.path.join(save_path, 'validation_results'))
        except:
            c_index_val = 0.
            print("Error calculating C-index for validation set")

    print(nnet_model.vy)
    if not is_early_stopping:
        torch.save(nnet_model.state_dict(), os.path.join(save_path, 'model_params_vae.pth'))
