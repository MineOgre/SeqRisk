import os

import pandas as pd
from lifelines import CoxPHFitter
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler

import torch
import numpy as np
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch


class _RepeatSampler(object):
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class HensmanDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    Dataloader when using minibatching with Stochastic Variational Inference.

    """

    def __init__(self, dataset, batch_sampler, num_workers):
        super().__init__(dataset, batch_sampler=_RepeatSampler(batch_sampler), num_workers=num_workers)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class SubjectSampler(Sampler):
    """
    Perform individual-wise sampling
    
    """

    def __init__(self, data_source, P, T):
        super(SubjectSampler, self).__init__(data_source)
        self.data_source = data_source
        self.P = P
        self.T = T

    def __iter__(self):
        r = np.arange(self.P)
        np.random.shuffle(r)
        list_of_lists = list(map(lambda x: [i for i in range(self.T * x, self.T * (x + 1))], r))
        res = list(itertools.chain.from_iterable(list_of_lists))
        return iter(res)

    def __len__(self):
        return len(self.data_source)


class VaryingLengthSubjectSampler(Sampler):
    """
    Perform individual-wise sampling when individuals have varying number of temporal samples.

    """

    def __init__(self, data_source, id_covariate):
        super(VaryingLengthSubjectSampler, self).__init__()
        self.data_source = data_source
        self.id_covariate = id_covariate

        def f(x):
            try:
                return int(x['label'][id_covariate].item())
            except IndexError:
                print(f"Error with sample: {x}")
                print(f"Label shape: {x['label'].shape}, id_covariate: {id_covariate}")
                raise

        l = []
        for sample in data_source:
            l.append(f(sample))
        # try:
        #     for sample in data_source:
        #         l.append(f(sample))
        # except:
        #     pass

        self.P = len(set(l))
        self.start_indices = [l.index(x) for x in list(OrderedDict.fromkeys(l))]
        self.end_indices = self.start_indices[1:] + [len(data_source)]

    def __iter__(self):
        r = np.arange(self.P)
        np.random.shuffle(r)
        list_of_lists = list(map(lambda x: [(i, x) for i in range(self.start_indices[x], self.end_indices[x])], r))
        res = iter(itertools.chain.from_iterable(list_of_lists))
        return iter(res)

    def __len__(self):
        return self.P


class VaryingLengthBatchSampler(BatchSampler):
    """
    Perform batch sampling when individuals have varying number of temporal samples.
    
    """

    def __init__(self, sampler, batch_size):
        super(VaryingLengthBatchSampler, self).__init__(sampler, batch_size, False)
        assert isinstance(sampler, VaryingLengthSubjectSampler)
        self.sampler = sampler
        self.batch_size = batch_size

    #__len__ defined by the superclass

    def __iter__(self):
        batch = []
        batch_subjects = set()
        for idx, subj in self.sampler:
            if subj not in batch_subjects:
                if len(batch_subjects) == self.batch_size:
                    yield batch
                    batch = []
                    batch_subjects.clear()
                batch_subjects.add(subj)
            batch.append(idx)
        yield batch


def batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x,
                            test_x, mu, zt_list, id_covariate, eps):
    """
    Perform batch predictions when individuals have varying number of temporal samples.
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = prediction_x.shape[1]
    M = zt_list[0].shape[0]

    I_M = torch.eye(M, dtype=torch.double).to(device)

    if isinstance(covar_module0, list):
        K0xz = torch.zeros(latent_dim, prediction_x.shape[0], M).double().to(device)
        K0zz = torch.zeros(latent_dim, M, M).double().to(device)
        K0Xz = torch.zeros(latent_dim, test_x.shape[0], M).double().to(device)

        for i in range(latent_dim):
            covar_module0[i].eval()
            covar_module1[i].eval()
            likelihoods[i].eval()
            z = zt_list[i].to(device)

            K0xz[i] = covar_module0[i](prediction_x, z).evaluate()
            K0zz[i] = covar_module0[i](z, z).evaluate()
            K0Xz[i] = covar_module0[i](test_x, z).evaluate()

    else:
        covar_module0.eval()
        covar_module1.eval()
        likelihoods.eval()

        K0xz = covar_module0(prediction_x, zt_list).evaluate()
        K0zz = covar_module0(zt_list, zt_list).evaluate()
        K0Xz = covar_module0(test_x, zt_list).evaluate()

    K0zz = K0zz + eps * I_M
    K0zx = K0xz.transpose(-1, -2)

    iB_st_list = []
    H = K0zz
    subjects = torch.unique(prediction_x[:, id_covariate]).tolist()
    iB_mu = torch.zeros(latent_dim, prediction_x.shape[0], 1, dtype=torch.double).to(device)
    for s in subjects:
        indices = prediction_x[:, id_covariate] == s
        x_st = prediction_x[indices]
        T = x_st.shape[0]
        I_T = torch.eye(T, dtype=torch.double).to(device)

        if isinstance(covar_module0, list):
            B_st = torch.zeros(latent_dim, T, T, dtype=torch.double).to(device)
            for i in range(latent_dim):
                B_st[i] = covar_module1[i](x_st, x_st).evaluate() + I_T * likelihoods[i].noise_covar.noise
        else:
            stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=0)
            B_st = covar_module1(stacked_x_st, stacked_x_st).evaluate() + I_T * likelihoods.noise_covar.noise.unsqueeze(
                dim=2)

        LB_st = torch.linalg.cholesky(B_st)
        iB_st = torch.cholesky_solve(I_T, LB_st)
        K0xz_st = K0xz[:, indices]
        K0zx_st = K0xz_st.transpose(-1, -2)
        iB_K0xz = torch.matmul(iB_st, K0xz_st)
        K0zx_iB_K0xz = torch.matmul(K0zx_st, iB_K0xz)
        H = H + K0zx_iB_K0xz
        iB_mu[:, indices] = torch.matmul(iB_st, mu[indices].T.unsqueeze(dim=2))
        iB_st_list.append(iB_st)

    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.linalg.solve(H, torch.matmul(K0zx, iB_mu)))
    iB_K0xz_iH_K0zx_iB_mu = torch.zeros(latent_dim, prediction_x.shape[0], 1, dtype=torch.double).to(device)
    for i, s in enumerate(subjects):
        indices = prediction_x[:, id_covariate] == s
        iB_K0xz_iH_K0zx_iB_mu[:, indices] = torch.matmul(iB_st_list[i], K0xz_iH_K0zx_iB_mu_st[:, indices])
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu

    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.linalg.solve(K0zz, torch.matmul(K0zx, mu_tilde)))

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(prediction_x[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(latent_dim, test_x.shape[0], 1, dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s

        if isinstance(covar_module0, list):
            K1Xx = torch.zeros(latent_dim, test_x[indices].shape[0], np.sum(mask)).double().to(device)
            for i in range(latent_dim):
                K1Xx[i] = covar_module1[i](test_x[indices], prediction_x[mask]).evaluate()
        else:
            stacked_test_x_indices = torch.stack([test_x[indices] for i in range(latent_dim)], dim=0)
            stacked_prediction_x_mask = torch.stack([prediction_x[mask] for i in range(latent_dim)], dim=0)
            K1Xx = covar_module1(stacked_test_x_indices, stacked_prediction_x_mask).evaluate()
        K1Xx_mu_tilde[:, indices] = torch.matmul(K1Xx, mu_tilde[:, mask])

    Z_pred = (K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde).squeeze(dim=2).T

    return Z_pred


def batch_predict(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, mu,
                  zt_list, P, T, id_covariate, eps):
    """
    Perform batch-wise predictions
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = prediction_x.shape[1]
    M = zt_list[0].shape[0]
    I_M = torch.eye(M, dtype=torch.double).to(device)
    I_T = torch.eye(T, dtype=torch.double).to(device)

    x_st = torch.reshape(prediction_x, [P, T, Q])

    mu = mu.T
    mu_st = torch.reshape(mu, [latent_dim, P, T, 1])

    if isinstance(covar_module0, list):
        K0xz = torch.zeros(latent_dim, P * T, M).double().to(device)
        K0zz = torch.zeros(latent_dim, M, M).double().to(device)
        B_st = torch.zeros(latent_dim, P, T, T).double().to(device)
        K0Xz = torch.zeros(latent_dim, test_x.shape[0], M).double().to(device)

        for i in range(latent_dim):
            covar_module0[i].eval()
            covar_module1[i].eval()
            likelihoods[i].eval()
            z = zt_list[i].to(device)

            K0xz[i] = covar_module0[i](prediction_x, z).evaluate()
            K0zz[i] = covar_module0[i](z, z).evaluate()
            B_st[i] = covar_module1[i](x_st, x_st).evaluate() + I_T * likelihoods[i].noise_covar.noise
            K0Xz[i] = covar_module0[i](test_x, z).evaluate()

    else:
        covar_module0.eval()
        covar_module1.eval()
        likelihoods.eval()

        stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)

        K0xz = covar_module0(prediction_x, zt_list).evaluate()
        K0zz = covar_module0(zt_list, zt_list).evaluate()
        B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + I_T * likelihoods.noise_covar.noise.unsqueeze(
            dim=2)).transpose(0, 1)
        K0Xz = covar_module0(test_x, zt_list).evaluate()

    K0zz = K0zz + eps * I_M
    LB_st = torch.linalg.cholesky(B_st)
    iB_st = torch.cholesky_solve(I_T, LB_st)
    K0xz_st = torch.reshape(K0xz, [latent_dim, P, T, M])
    K0zx_st = K0xz_st.transpose(-1, -2)
    K0zx = K0xz.transpose(-1, -2)

    iB_K0xz = torch.matmul(iB_st, K0xz_st)
    K0zx_iB_K0xz = torch.matmul(K0zx, torch.reshape(iB_K0xz, [latent_dim, P * T, M]))
    H = K0zz + K0zx_iB_K0xz
    iB_mu = torch.matmul(iB_st, mu_st).view(latent_dim, -1, 1)
    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.linalg.solve(H, torch.matmul(K0zx, iB_mu))).reshape(latent_dim, P,
                                                                                                         T, -1)
    iB_K0xz_iH_K0zx_iB_mu = torch.matmul(iB_st, K0xz_iH_K0zx_iB_mu_st).view(latent_dim, -1, 1)
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu
    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.linalg.solve(K0zz, torch.matmul(K0zx, mu_tilde)))

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(prediction_x[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(latent_dim, test_x.shape[0], 1, dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s

        if isinstance(covar_module0, list):
            K1Xx = torch.zeros(latent_dim, test_x[indices].shape[0], np.sum(mask)).double().to(device)
            for i in range(latent_dim):
                K1Xx[i] = covar_module1[i](test_x[indices], prediction_x[mask]).evaluate()
        else:
            stacked_test_x_indices = torch.stack([test_x[indices] for i in range(latent_dim)], dim=0)
            stacked_prediction_x_mask = torch.stack([prediction_x[mask] for i in range(latent_dim)], dim=0)
            K1Xx = covar_module1(stacked_test_x_indices, stacked_prediction_x_mask).evaluate()

        K1Xx_mu_tilde[:, indices] = torch.matmul(K1Xx, mu_tilde[:, mask])

    Z_pred = (K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde).squeeze(dim=2).T

    return Z_pred


def predict(covar_module0, covar_module1, likelihood, train_xt, test_x, mu, z, P, T, id_covariate, eps):
    """
    Helper function to perform predictions.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = train_xt.shape[1]
    M = z.shape[0]
    I_M = torch.eye(M, dtype=torch.double).to(device)
    I_T = torch.eye(T, dtype=torch.double).to(device)

    x_st = torch.reshape(train_xt, [P, T, Q])
    mu_st = torch.reshape(mu, [P, T, 1])

    K0xz = covar_module0(train_xt, z).evaluate()
    K0zz = covar_module0(z, z).evaluate() + eps * I_M
    K1_st = covar_module1(x_st, x_st).evaluate()
    K0Xz = covar_module0(test_x, z).evaluate()

    B_st = K1_st + I_T * likelihood.noise_covar.noise
    LB_st = torch.linalg.cholesky(B_st)
    iB_st = torch.cholesky_solve(I_T, LB_st)
    K0xz_st = torch.reshape(K0xz, [P, T, M])
    K0zx_st = K0xz_st.transpose(-1, -2)

    iB_K0xz = torch.matmul(iB_st, K0xz_st)
    K0zx_iB_K0xz = torch.matmul(K0xz.T, torch.reshape(iB_K0xz, [P * T, M]))
    H = K0zz + K0zx_iB_K0xz

    iB_mu = torch.matmul(iB_st, mu_st).view(-1)
    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz,
                                         torch.linalg.solve(H, torch.matmul(K0xz.T, iB_mu).unsqueeze(dim=1))).reshape(P,
                                                                                                                      T,
                                                                                                                      -1)
    iB_K0xz_iH_K0zx_iB_mu = torch.matmul(iB_st, K0xz_iH_K0zx_iB_mu_st).view(-1)
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu
    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.linalg.solve(K0zz, torch.matmul(K0xz.T, mu_tilde).unsqueeze(
        dim=1))).squeeze()

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(train_xt[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(test_x.shape[0], dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s
        K1Xx = covar_module1(test_x[indices], train_xt[mask]).evaluate()
        K1Xx_mu_tilde[indices] = torch.matmul(K1Xx, mu_tilde[mask])

    Z_pred = K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde

    return Z_pred


def get_last_N_obs(group, N, time_covariate):
    sorted_group = group.sort_values(time_covariate).tail(N)
    new_row = {}

    return sorted_group.index.values


def cox_loss_last_n(N, train_x, id_covariate, time_covariate, event_covariate):
    # Convert train_x to a DataFrame
    df_train_x = pd.DataFrame(train_x.cpu().numpy())

    # Exclude rows where event_covariate is -1
    df_train_x = df_train_x[df_train_x[event_covariate] != -1]

    # Get the column name for id_covariate
    id_column = df_train_x.columns[id_covariate]

    # Apply the groupby and get_last_N_obs operations
    survival_indexes = df_train_x.groupby(id_column).apply(get_last_N_obs, N=N,
                                                           time_covariate=time_covariate).reset_index(drop=True)

    return survival_indexes


import torch
from lifelines.utils import concordance_index


def concordance_index_long(prediction, time_survival, death, eval_time):
    '''
        This is a cause-specific c(t)-index
        - prediction      : risk at eval_time (higher --> more risky)
        - time_survival   : survival/censoring eval_time
        - death           :
            > 1: death
            > 0: censored (including death from other cause)
        - eval_time            : eval_time of evaluation (eval_time-horizon when evaluating C-index)
    '''
    N = len(prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    prediction_cpu = prediction.cpu().detach().numpy()
    for i in range(N):
        A[i, np.where(time_survival[i] < time_survival)] = 1
        Q[i, np.where(prediction_cpu[i] > prediction_cpu)] = 1

        if (time_survival[i] <= eval_time and death[i] == 1):
            N_t[i, :] = 1

    # A[:, 0] = 0
    # A[np.where(time_survival < time_survival[0]), 0] = 1
    # A[0,0] = 0

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    # print(f"NUMERATOR: {Num}")
    # print(f"DENOMINATOR: {Den}")

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result, Den


def calculate_risk_score(nnet_model, mu, label, surv_covariate, save_path=None):
    """
    Calculate the risk and concordance index.

    Parameters:
    - nnet_model: The neural network model.
    - mu: The mu tensor.
    - label: The label data.
    - mask: The mask data.

    Returns:
    - risk: Calculated risk.
    - last_indexes: Last indexes of survival.
    - concordance_indx: Concordance index.
    - E: Event data.
    - time_data: eval_time data.
    """

    resulting_tensor, structured_mu, E, time_data, src_mask, last_meas_time = nnet_model.get_survival_data(mu, label,
                                                                                                           surv_covariate)
    _EPSILON = 1e-08
    if 'survival' in nnet_model.model_type:

        # if nnet_model.transformer and 'long' not in nnet_model.model_type:
        #     risk = nnet_model.risk_forward(structured_mu,src_mask)
        #     concordance_indx = concordance_index(
        #         time_data.detach().cpu().numpy(),
        #         -torch.exp(risk).detach().cpu().numpy(),
        #         E.detach().cpu().numpy()
        #     )
        if 'long' in nnet_model.model_type:
            if save_path is not None:
                # Create folder if it does not exist
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            # Extract IDs and event information
            ids = label[:, nnet_model.id_covariate]
            events = label[:, nnet_model.event_covariate]

            # Prepare a tensor to hold the updated event information
            updated_events = torch.zeros_like(events)

            # Update the event information for each ID
            unique_ids = ids.unique()
            for uid in unique_ids:
                # Find the last index of the current ID
                last_idx = (ids == uid).nonzero(as_tuple=True)[0][-1]
                # Update all rows with the event from the last row of the current ID
                updated_events[ids == uid] = events[last_idx]

            # Replace the event information in the original data
            label[:, nnet_model.event_covariate] = updated_events

            num_Event = 1

            risk_all = {}
            for k in range(num_Event):
                risk_all[k] = np.zeros([structured_mu.shape[1], len(nnet_model.pred_time), len(nnet_model.eval_time)])

            last_meas, structured_mu, E, time_data, src_mask, last_meas_time = (
                nnet_model.get_survival_data(mu,
                                             label,
                                             surv_covariate))
            for p, p_time in enumerate(nnet_model.pred_time):
                ### PREDICTION
                pred_horizon = int(p_time)
                # last_meas, structured_mu, E, time_data, src_mask, last_meas_time = (
                #     nnet_model.get_survival_data(mu[label[:,nnet_model.start_time_covariate]<=pred_horizon],
                #                                  label[label[:,nnet_model.start_time_covariate]<=pred_horizon],
                #                                  surv_covariate[label[:,nnet_model.start_time_covariate]<=pred_horizon]))

                # pred = nnet_model.risk_forward(structured_mu, src_mask, last_meas)
                if 'transformer' in nnet_model.model_type:
                    pred = nnet_model.risk_forward(structured_mu, src_mask)
                else:
                    pred = nnet_model.hazard_net(last_meas)
                E_np = E.detach().cpu().numpy()
                second_column = last_meas_time.detach().cpu().numpy() + time_data.detach().cpu().numpy()
                pred = torch.nn.Softmax(dim=1)(pred)
                # plot_event_probs(save_path, full_out=pred.detach().cpu().numpy()[:, None, :],
                #                  event_time=np.column_stack((E_np, second_column)))

                val_result1 = np.zeros([num_Event, len(nnet_model.eval_time)])
                denom = np.zeros([num_Event, len(nnet_model.eval_time)])

                for t, t_time in enumerate(nnet_model.eval_time):
                    eval_horizon = int(t_time) + pred_horizon  # if eval_horizon >= num_Category, output the maximum...
                    # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                    risk = torch.sum(pred[:, pred_horizon:(eval_horizon + 1)], axis=1,
                                     keepdim=True)  # risk score until eval_time
                    # risk = risk / (torch.sum(torch.sum(pred[:, pred_horizon:], axis=1), axis=1,
                    #                       keepdim=True) + _EPSILON)  # conditioniong on t > t_pred    Multiple events
                    risk = risk / (torch.sum(pred[:, pred_horizon:], axis=1, keepdim=True) + _EPSILON)  # one event

                    for k in range(num_Event):
                        risk_all[k][:, p, t] = risk[:, k].detach().cpu().numpy()
                        time_survival = (last_meas_time + time_data).detach().cpu().numpy()
                        Death = E.detach().cpu().numpy()
                        val_result1[k, t], denom[k, t] = concordance_index_long(risk[:, k], time_survival, Death,
                                                                                eval_horizon)

                if p == 0:
                    val_final1 = val_result1
                    val_denom = denom
                else:
                    val_final1 = np.append(val_final1, val_result1, axis=0)
                    val_denom = np.append(val_denom, denom, axis=0)

            row_header = []
            for p_time in nnet_model.pred_time:
                for t in range(num_Event):
                    row_header.append('pred_time {}: event_{}'.format(p_time, k + 1))

            col_header = []
            for t_time in nnet_model.eval_time:
                col_header.append('eval_time {}'.format(t_time))

            # c-index result
            df1 = pd.DataFrame(val_final1, index=row_header, columns=col_header)
            df2 = pd.DataFrame(val_denom, index=row_header, columns=col_header)

            print(df1)
            print(df2)
            concordance_indx = np.mean(val_final1[val_final1 != -1])
            # Save val_final1 and val_denom to save_path location

            np.savetxt(os.path.join(save_path, 'concordance_all.csv'), val_final1, delimiter=',')
            np.savetxt(os.path.join(save_path, 'denom_all.csv'), val_denom, delimiter=',')
            # np.save(val_denom, save_path + 'denom_all.csv')
        else:
            if nnet_model.transformer:
                risk = nnet_model.risk_forward(structured_mu, src_mask)
            else:
                risk = nnet_model.hazard_net(resulting_tensor)
            concordance_indx = concordance_index(
                time_data.detach().cpu().numpy(),
                -torch.exp(risk).detach().cpu().numpy(),
                E.detach().cpu().numpy()
            )
    else:
        # Convert tensors to pandas DataFrame
        # Assuming resulting_tensor contains the covariate data
        concordance_indx = nnet_model.coxfitter.score(df, scoring_method="concordance_index")

    print('Concordance index: ' + str(concordance_indx))

    # return risk, last_indexes, concordance_indx, E, time_data
    # return concordance_indx, df
    return concordance_indx


# def c_index_new(Prediction, Time_survival, Death):
#     '''
#         This is a cause-specific c(t)-index
#         - Prediction      : risk at Time (higher --> more risky)
#         - Time_survival   : survival/censoring eval_time
#         - Death           :
#             > 1: death
#             > 0: censored (including death from other cause)
#         - Time            : eval_time of evaluation (eval_time-horizon when evaluating C-index)
#     '''
#     N = len(Prediction)
#     A = np.zeros((N, N))
#     Q = np.zeros((N, N))
#     N_t = np.zeros((N, N))
#     Num = 0
#     Den = 0
#     for i in range(N):
#         A[i, np.where(Time_survival[i] < Time_survival)] = 1
#         Q[i, np.where(Prediction[i] > Prediction)] = 1
#         # T[i, np.where(Time_survival[i] < Time_survival)] = 1
#         if Death[i] == 1:
#             N_t[i, :] = 1
#         # if (Death[i]==1):
#         #     N_t[i,:] = 1
#
#     A[:, 0] = 0
#     A[np.where(Time_survival < Time_survival[0]), 0] = 1
#     A[0, 0] = 0
#
#     Num = np.sum(((A) * N_t) * Q)
#     Den = np.sum((A) * N_t)
#
#     # print(f"A: {np.sum(A)}")
#     print(f"NUMERATOR: {Num}")
#     print(f"DENOMINATOR: {Den}")
#     print(f"N: {N_t[:, 0].sum()}")
#
#     if Num == 0 and Den == 0:
#         result = 0.5  # not able to compute c-index!
#     else:
#         result = float(Num / Den)
#
#     print(f"c_index RESULT: {result}")
#     return result
#



def c_index(Prediction, Time_survival, last_meas_time, Death, Eval_Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring eval_time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : eval_time of evaluation (eval_time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
        # T[i, np.where(Time_survival[i] < Time_survival)] = 1
        if ((Time_survival[i] + last_meas_time[i]) < Eval_Time) and (Death[i] == 1):
            N_t[i, :] = 1
        # if (Death[i]==1):
        #     N_t[i,:] = 1

    A[:, 0] = 0
    A[np.where(Time_survival < Time_survival[0]), 0] = 1
    A[0, 0] = 0

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    # print(f"A: {np.sum(A)}")
    # print(f"NUMERATOR: {Num}")
    # print(f"DENOMINATOR: {Den}")
    # print(f"N: {N_t[:, 0].sum()}")

    if Num == 0 and Den == 0:
        result = 0.5  # not able to compute c-index!
    else:
        result = float(Num / Den)

    # print(f"c_index RESULT: {result}")
    return result


def c_index_truncated(Prediction, Time_survival, last_meas_time, Death, Pred_time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring eval_time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : eval_time of evaluation (eval_time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where((Time_survival[i] + last_meas_time[i]) < (last_meas_time+Time_survival))] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
        # T[i, np.where(Time_survival[i] < Time_survival)] = 1
        if ((Time_survival[i] + last_meas_time[i]) <= Pred_time) and (Death[i] == 1):
            N_t[i, :] = 1
        # if (Death[i]==1):
        #     N_t[i,:] = 1

    A[:, 0] = 0
    A[np.where(Time_survival < Time_survival[0]), 0] = 1
    A[0, 0] = 0

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    # print(f"A: {np.sum(A)}")
    # print(f"NUMERATOR: {Num}")
    # print(f"DENOMINATOR: {Den}")
    # print(f"N: {N_t[:, 0].sum()}")

    if Num == 0 and Den == 0:
        result = 0.5  # not able to compute c-index!
    else:
        result = float(Num / Den)

    # print(f"c_index RESULT: {result}")
    return result,N_t[:, 0].sum(), Num, Den


# def c_index_death(Prediction, Time_survival, Death):
#     '''
#         This is a cause-specific c(t)-index
#         - Prediction      : risk at Time (higher --> more risky)
#         - Time_survival   : survival/censoring eval_time
#         - Death           :
#             > 1: death
#             > 0: censored (including death from other cause)
#         - Time            : eval_time of evaluation (eval_time-horizon when evaluating C-index)
#     '''
#     N = len(Prediction)
#     A = np.zeros((N, N))
#     Q = np.zeros((N, N))
#     N_t = np.zeros((N, N))
#     Num = 0
#     Den = 0
#     for i in range(N):
#         A[i, np.where(Time_survival[i] < Time_survival)] = 1
#         Q[i, np.where(Prediction[i] > Prediction)] = 1
#         if (Death[i] == 1):
#             N_t[i, :] = 1
#         # if (Death[i]==1):
#         #     N_t[i,:] = 1
#
#     A[:, 0] = 0
#     A[np.where(Time_survival < Time_survival[0]), 0] = 1
#     A[0, 0] = 0
#
#     Num = np.sum(((A) * N_t) * Q)
#     Den = np.sum((A) * N_t)
#
#     # print(f"A: {np.sum(A)}")
#     print(f"NUMERATOR: {Num}")
#     print(f"DENOMINATOR: {Den}")
#     print(f"N: {N}")
#
#     if Num == 0 and Den == 0:
#         result = 0.5  # not able to compute c-index!
#     else:
#         result = float(Num / Den)
#
#     print(f"c_index_death RESULT: {result}")
#
#     return result


#
# def c_index_none(Prediction, Time_survival):
#     '''
#         This is a cause-specific c(t)-index
#         - Prediction      : risk at Time (higher --> more risky)
#         - Time_survival   : survival/censoring eval_time
#         - Death           :
#             > 1: death
#             > 0: censored (including death from other cause)
#         - Time            : eval_time of evaluation (eval_time-horizon when evaluating C-index)
#     '''
#     N = len(Prediction)
#     A = np.zeros((N,N))
#     Q = np.zeros((N,N))
#     N_t = np.ones((N,N))
#     Num = 0
#     Den = 0
#     for i in range(N):
#         A[i, np.where(Time_survival[i] < Time_survival)] = 1
#         Q[i, np.where(Prediction[i] > Prediction)] = 1
#         # if (Death[i]==1):
#         #     N_t[i,:] = 1
#         # if (Death[i]==1):
#         #     N_t[i,:] = 1
#
#     A[:, 0] = 0
#     A[np.where(Time_survival < Time_survival[0]), 0] = 1
#     A[0,0] = 0
#
#     Num  = np.sum(((A)*N_t)*Q)
#     Den  = np.sum((A)*N_t)
#
#
#     # print(f"A: {np.sum(A)}")
#     print(f"NUMERATOR: {Num}")
#     print(f"DENOMINATOR: {Den}")
#
#     if Num == 0 and Den == 0:
#         result = 0.5 # not able to compute c-index!
#     else:
#         result = float(Num/Den)
#
#     return result

def cut_data_by_time(mu, label, surv_covariate, pred_time, start_time_covariate, stop_time_covariate,
                     id_covariate, event_covariate):
    """
    Remove the data points whose start eval_time is after the prediction eval_time and
    last stop eval_time is before the prediction eval_time.
    returns new mu, label and surv_covariate tensors.
    """

    # Get the start eval_time and stop eval_time from the label tensor
    stop_time = label[:, stop_time_covariate]

    # Get the unique IDs whos maximum stop eval_time is greater than the prediction eval_time
    # unique_ids = label[:, id_covariate].unique()
    # valid_ids = []
    # for uid in unique_ids:
    #     mask = label[:, id_covariate] == uid
    #     if torch.max(stop_time[mask]) <= pred_time:
    #         valid_ids.append(uid)

    # Create a mask to filter the data points
    mask = torch.ones(mu.shape[0], dtype=torch.bool)
    # for uid in valid_ids:
    #     mask = mask | (label[:, id_covariate] == uid)
    #
    # Filter the data points
    new_mu = mu[mask]
    new_label = label[mask]
    new_surv_covariate = surv_covariate[mask]

    # Get the max time_covariate values for each ID
    unique_ids = label[:, id_covariate].unique()
    max_time_covariate = []
    event_covariate_list = []
    for uid in unique_ids:
        mask = label[:, id_covariate] == uid
        max_time_covariate.append(torch.max(label[mask, stop_time_covariate]))
        event_covariate_list.append(torch.max(label[mask, event_covariate]))

    # Remove the data points whose start eval_time is after the prediction eval_time
    mask = new_label[:, start_time_covariate] <= pred_time
    new_mu = new_mu[mask]
    new_label = new_label[mask]
    new_surv_covariate = new_surv_covariate[mask]

    # Update the stop_time_covariate value of the last row  for each ID according to max_time_covariate
    for i, uid in enumerate(unique_ids):
        mask = new_label[:, id_covariate] == uid
        reversed_mask = mask.tolist()[::-1]
        try:
            reversed_index = reversed_mask.index(True)
        except:
            continue
        original_index = len(mask) - 1 - reversed_index
        new_label[original_index, stop_time_covariate] = max_time_covariate[i]
        new_label[original_index, event_covariate] = event_covariate_list[i]

    return new_mu, new_label, new_surv_covariate


def calculate_risk_score_pred(nnet_model, mu, label, surv_covariate, save_path=None):
    """
    Calculate the risk and concordance index.

    Parameters:
    - nnet_model: The neural network model.
    - mu: The mu tensor.
    - label: The label data.
    - mask: The mask data.

    Returns:
    - risk: Calculated risk.
    - last_indexes: Last indexes of survival.
    - concordance_indx: Concordance index.
    - E: Event data.
    - time_data: eval_time data.
    """

    _EPSILON = 1e-08
    c_index_pred_eval = np.zeros((len(nnet_model.pred_time), len(nnet_model.eval_time)))
    N_all = np.zeros((len(nnet_model.pred_time), len(nnet_model.eval_time)))
    Num_all = np.zeros((len(nnet_model.pred_time), len(nnet_model.eval_time)))
    Den_all = np.zeros((len(nnet_model.pred_time), len(nnet_model.eval_time)))

    if 'survival' in nnet_model.model_type:
        concordance_index_lst = []
        for p, pred_time in enumerate(nnet_model.pred_time):
            # print("PRED TIME: ", pred_time)
            new_mu, new_label, new_surv_covariate = cut_data_by_time(mu, label, surv_covariate, pred_time,
                                                                     nnet_model.start_time_covariate,
                                                                     nnet_model.time_covariate,
                                                                    nnet_model.id_covariate,
                                                                     nnet_model.event_covariate)

            resulting_tensor, structured_mu, E, time_data, src_mask, last_meas_time = nnet_model.get_survival_data(
                new_mu, new_label, new_surv_covariate)
            if nnet_model.transformer:
                risk = nnet_model.risk_forward(structured_mu, src_mask)
            else:
                risk = nnet_model.hazard_net(resulting_tensor)
            concordance_indx = concordance_index(
                time_data.detach().cpu().numpy(),
                -torch.exp(risk).detach().cpu().numpy(),
                E.detach().cpu().numpy()
            )
            # concordance_index_truncated = c_index(torch.exp(risk), time_data,last_meas_time, E, pred_time)
            # print(f"Concordance Index: {concordance_index_truncated} at PRED TIME: {pred_time}")
            for e, eval_time in enumerate(nnet_model.eval_time):
                c_index_, N_all[p,e], Num_all[p,e], Den_all[p,e] = c_index_truncated(torch.exp(risk), time_data,last_meas_time, E, pred_time+eval_time)
                # print(f"Concordance Index: {c_index_} at PRED TIME: {pred_time} and EVAL TIME: {eval_time}")
                c_index_pred_eval[p, e] = c_index_
            # print('Concordance index: ' + str(concordance_indx))
            concordance_index_lst.append(concordance_indx)
    else:
        # Convert tensors to pandas DataFrame
        # Assuming resulting_tensor contains the covariate data
        concordance_indx = 0.  # nnet_model.coxfitter.score(df, scoring_method="concordance_index")

    # print('Concordance index: ' + str(concordance_indx))
    concordance_index_lst = np.array(concordance_index_lst)
    pd_concordance_index = pd.DataFrame(concordance_index_lst, index=nnet_model.pred_time,
                                        columns=['Concordance Index'])
    pd_pred_eval = pd.DataFrame(c_index_pred_eval, index=nnet_model.pred_time, columns=nnet_model.eval_time)
    ### write the dataframe to csv
    pd_concordance_index.to_csv(os.path.join(save_path, 'concordance_index_test_pred.csv'))
    print(pd_concordance_index)
    print(pd_pred_eval)
    # return risk, last_indexes, concordance_indx, E, time_data
    # return concordance_indx, df
    return pd_concordance_index


def train_COX_on_latent(nnet_model, mu, label):
    """
    Calculate the risk and concordance index.

    Parameters:
    - nnet_model: The neural network model.
    - mu: The mu tensor.
    - label: The label data.
    - mask: The mask data.

    Returns:
    - risk: Calculated risk.
    - last_indexes: Last indexes of survival.
    - concordance_indx: Concordance index.
    - E: Event data.
    - time_data: eval_time data.
    """

    survival_indexes = cox_loss_last_n(
        nnet_model.last_N, label, nnet_model.id_covariate,
        nnet_model.time_covariate, nnet_model.event_covariate
    )

    resulting_rows = []
    for index_list in survival_indexes:
        rows = mu[index_list]
        new_row = torch.cat(tuple(rows), dim=0)
        resulting_rows.append(new_row)

    last_indexes = [index_list[-1] for index_list in survival_indexes]
    E = label[last_indexes, nnet_model.event_covariate]
    time_data = label[last_indexes, nnet_model.time_covariate] - label[
        last_indexes, nnet_model.start_time_covariate]  ###TODO: Hard coded start_obs

    resulting_tensor = torch.stack(resulting_rows, dim=0)

    # Convert tensors to pandas DataFrame
    # Assuming resulting_tensor contains the covariate data
    covariate_columns = [f"covariate_{i + 1}" for i in range(resulting_tensor.shape[1])]
    df = pd.DataFrame(resulting_tensor.cpu().detach().numpy(), columns=covariate_columns)

    # Add event and eval_time columns
    df['event'] = E.numpy()
    df['eval_time'] = time_data.numpy()
    coxfitter = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    coxfitter = coxfitter.fit(df, event_col="event", duration_col="eval_time",
                              show_progress=False)

    return coxfitter


def plot_event_probs(results_path, full_out=None, event_time=None, epoch=0):
    if full_out is not None:

        np.random.seed(10)

        num_subjects = full_out.shape[0]
        time_bins = full_out.shape[2]

        # Step 1: Randomly select 10 subjects
        num_to_select = np.min([60, num_subjects])
        indices = np.random.permutation(num_subjects)[:num_to_select]

        # Step 2: Extract the data for the selected subjects
        selected_data = full_out[indices].squeeze()
        if event_time is not None:
            selected_events = event_time[np.array(indices), :]  # Extract corresponding event data

        # Step 3: Plotting each subject in a separate subplot
        fig, axs = plt.subplots(nrows=num_to_select, figsize=(10, num_to_select * 2))  # Adjust the size as needed

        time_points = np.arange(time_bins)
        for i, (ax, subject_data) in enumerate(zip(axs, selected_data)):
            ax.plot(time_points, subject_data, marker='o', linestyle='-')  # You can customize the plot style
            ax.set_title(f'Subject {indices[i].item()}')
            ax.set_xlabel('Time Bins')
            ax.set_ylabel('Probability')
            ax.grid(True)

            if event_time is not None:
                # Plot vertical lines for event/censored times
                event_or_censor = selected_events[i, 0]  # 1 for event, 0 for censored
                time_of_event = selected_events[i, 1]
                if event_or_censor == 1:
                    ax.axvline(x=time_of_event, color='red', label='Event', linestyle='--')
                else:
                    ax.axvline(x=time_of_event, color='blue', label='Censored', linestyle=':')

            ax.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if epoch != 0:
            # Save the entire figure
            plt.savefig(f'{results_path}/training_probability_values_{epoch}.pdf')
        else:
            plt.savefig(f'{results_path}/validation_probability_values.pdf')

        # Optionally display the plot
        plt.show()


def plot_training_values(net_train_loss_arr, recon_loss_arr, nll_loss_arr, kld_loss_arr, surv_loss_arr,
                         train_c_index_arr, results_path, full_out=None, event_time=None, epoch=0):
    """
    Plot the training values.

    Parameters:
    - net_train_loss_arr: The array of network training loss.
    - recon_loss_arr: The array of reconstruction loss.
    - nll_loss_arr: The array of negative log likelihood loss.
    - kld_loss_arr: The array of KL divergence loss.
    - surv_loss_arr: The array of survival loss.

    Returns:
    - None
    """

    # Define the number of subplots you want to create
    num_plots = 6
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

    # Plot each loss on a separate subplot
    axes[0].plot(net_train_loss_arr, label='Net Train Loss')
    axes[0].set_title('Net Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Net Train Loss')
    axes[0].legend()

    axes[1].plot(recon_loss_arr, label='Reconstruction Loss', color='orange')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].legend()

    axes[2].plot(nll_loss_arr, label='NLL Loss', color='green')
    axes[2].set_title('NLL Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('NLL Loss')
    axes[2].legend()

    axes[3].plot(kld_loss_arr, label='KLD Loss', color='red')
    axes[3].set_title('KLD Loss')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('KLD Loss')
    axes[3].legend()

    axes[4].plot(surv_loss_arr, label='Survival Loss', color='purple')
    axes[4].set_title('Survival Loss')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Survival Loss')
    axes[4].legend()

    axes[5].plot(train_c_index_arr, label='Traning C-Index', color='blue')
    axes[5].set_title('Traning C-Index')
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('Traning C-Index')
    axes[5].legend()

    # Adjust the layout so that all subplots fit into the figure neatly
    plt.tight_layout()
    plt.savefig(f'{results_path}/training_values.pdf')
    plt.show()
    plt.close()

    if full_out is not None:

        torch.manual_seed(10)

        # Example setup, ensure these are defined correctly in your actual code
        # full_out = torch.rand(100, 50)  # Simulated probability data for 100 subjects over 50 eval_time bins
        # Event_time = torch.tensor([[1, 10], [0, 15], ...])  # Simulated event/censoring times

        num_subjects = full_out.shape[0]
        time_bins = full_out.shape[1]

        # Step 1: Randomly select 10 subjects
        num_to_select = 60
        indices = torch.randperm(num_subjects)[:num_to_select]

        # Step 2: Extract the data for the selected subjects
        selected_data = full_out[indices].detach().cpu().numpy()
        selected_events = event_time.iloc[np.array(indices), :]  # Extract corresponding event data

        # Step 3: Plotting each subject in a separate subplot
        fig, axs = plt.subplots(nrows=num_to_select, figsize=(10, num_to_select * 2))  # Adjust the size as needed

        time_points = torch.arange(time_bins)
        for i, (ax, subject_data) in enumerate(zip(axs, selected_data)):
            ax.plot(time_points, subject_data, marker='o', linestyle='-')  # You can customize the plot style
            ax.set_title(f'Subject {indices[i].item()}')
            ax.set_xlabel('Time Bins')
            ax.set_ylabel('Probability')
            ax.grid(True)

            # Plot vertical lines for event/censored times
            event_or_censor = selected_events.iloc[i, 0]  # 1 for event, 0 for censored
            time_of_event = selected_events.iloc[i, 1]
            if event_or_censor == 1:
                ax.axvline(x=time_of_event, color='red', label='Event', linestyle='--')
            else:
                ax.axvline(x=time_of_event, color='blue', label='Censored', linestyle=':')

            ax.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the entire figure
        plt.savefig(f'{results_path}/training_probability_values_{epoch}.pdf')

        # Optionally display the plot
        plt.show()
        #
        #
        # # Assuming `full_out` is already a PyTorch tensor containing your data
        # num_subjects = full_out.shape[0]
        # time_bins = full_out.shape[1]
        #
        # # Step 1: Randomly select 10 subjects
        # num_to_select = 10
        # indices = torch.randperm(num_subjects)[:num_to_select]
        #
        # # Step 2: Extract the data for the selected subjects
        # selected_data = full_out[indices].detach().numpy()
        #
        # # Step 3: Plotting each subject in a separate subplot
        # fig, axs = plt.subplots(nrows=num_to_select, figsize=(10, 20))  # Adjust the size as needed
        #
        # time_points = torch.arange(time_bins)
        # for i, (ax, subject_data) in enumerate(zip(axs, selected_data)):
        #     ax.plot(time_points, subject_data, marker='o', linestyle='-')  # You can customize the plot style
        #     ax.set_title(f'Subject {indices[i].item()}')
        #     ax.set_xlabel('Time Bins')
        #     ax.set_ylabel('Probability')
        #     ax.grid(True)
        #
        # # Adjust layout to prevent overlap
        # plt.tight_layout()
        #
        # # Save the entire figure
        # plt.savefig(f'{results_path}/training_probability_values.pdf')
        #
        # # Optionally display the plot
        # plt.show()


def plot_validation_values(validation_results, results_path, validation_criteria):
    """
    Plot the validation values.
    """
    num_plots = len(validation_criteria)
    colors = ['red', 'green', 'orange', 'purple']
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

    for i, criterion in enumerate(validation_criteria):
        axes[i].plot(validation_results[i], label=criterion, color=colors[i])
        axes[i].set_title(criterion)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(criterion)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'{results_path}/validation_values.pdf')
    plt.show()
    plt.close()
