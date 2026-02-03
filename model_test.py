import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

import wandb
from VAE import cox_loss_last_n

from dataset_def import HealthMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDatasetConv, HUSCorogeneDataset
from utils import batch_predict, batch_predict_varying_T, calculate_risk_score, calculate_risk_score_pred


def predict_gp(kernel_component, full_kernel_inverse, z):
    """
    Function to compute predictive mean

    """
    mean = torch.matmul(torch.matmul(kernel_component, full_kernel_inverse), z)
    return mean

def MSE_test(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
             covar_module, likelihoods, results_path, latent_dim, prediction_x, prediction_mu):
    
    """
    Function to compute Mean Squared Error of test set.
    
    """
    print("Running tests with a test set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                              csv_file_label=csv_file_test_label,
                                              mask_file=test_mask_file, root_dir=data_source_path,
                                              transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor()) 

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))
            if(prediction_x.shape[0] > 6040):
                r = np.random.choice(prediction_x.shape[0], 6000, replace=False) + 40
                ind = np.concatenate((np.arange(40), r))
                prediction_x = prediction_x[ind]
                prediction_mu = prediction_mu[ind]

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            for i in range(0, latent_dim):
                covar_module[i].eval()
                likelihoods[i].eval()
                K1 = covar_module[i](prediction_x.to(device), prediction_x.to(device)).evaluate() \
                                     + likelihoods[i].noise * torch.eye(prediction_mu.shape[0]).to(device)
                LK1 = torch.cholesky(K1)
                iK1 = torch.cholesky_solve(torch.eye(prediction_mu.shape[0], dtype=torch.double).to(device), LK1).to(device)
                kernel_component = covar_module[i](test_x.to(device), prediction_x.to(device)).evaluate()
                pred_means = predict_gp(kernel_component, iK1, prediction_mu[:, i])
                Z_pred = torch.cat((Z_pred, pred_means.view(-1, 1)), 1)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            np.savetxt(os.path.join(results_path, 'result_error.csv'), pred_results)


def MSE_test_GPapprox(test_dataset, data_source_path, type_nnet, nnet_model,
                      covar_module0, covar_module1, likelihoods, results_path, latent_dim, prediction_x, prediction_mu,
                      zt_list, P, T, id_covariate, dataset_type, varying_T=False, is_early_stopping=False, num_workers=0):

    """
    Function to compute Mean Squared Error of test set with GP approximationö
    
    """

    print("Running tests with a test set")
    # dataset_type = 'HealthMNIST'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if type_nnet == 'conv':
    #     if dataset_type == 'HealthMNIST':
    #         test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
    #                                               csv_file_label=csv_file_test_label,
    #                                               mask_file=test_mask_file, root_dir=data_source_path,
    #                                               transform=transforms.ToTensor())
    #     elif dataset_type == 'RotatedMNIST':
    #         test_dataset = RotatedMNISTDatasetConv(data_file=csv_file_test_data,
    #                                                label_file=csv_file_test_label,
    #                                                mask_file=test_mask_file, root_dir=data_source_path,
    #                                                transform=transforms.ToTensor())
    #
    # elif type_nnet == 'simple':
    #     if dataset_type == 'HUSCorogene':
    #         test_dataset = HUSCorogeneDataset(csv_file_data=csv_file_test_data,
    #                                           csv_file_label=csv_file_test_label,
    #                                           mask_file=test_mask_file, root_dir=data_source_path)
    #     else:
    #         test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
    #                                       csv_file_label=csv_file_test_label,
    #                                       mask_file=test_mask_file, root_dir=data_source_path,
    #                                       transform=transforms.ToTensor())

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)
    concordance_indx = 0.0

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label'].to(device)
            surv_x = sample_batched['surv_covariates'].to(device)
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask'].to(device)
            data = data * mask.reshape(data.shape).to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            if len(nnet_model.pred_time):
                calculate_risk_score_pred(nnet_model, mu, label, surv_x,
                                                    # save_path=os.path.join(results_path,'test_results'))
                                                    save_path =results_path)
            # else:
            concordance_indx = calculate_risk_score(nnet_model, mu, label, surv_x,
                                                save_path =results_path)

            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)
            
            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            # wandb.log({"recon_loss": torch.mean(recon_loss),
            #            "recon_loss_GP": torch.mean(recon_loss_GP),
            #            "concordance_index": concordance_indx})
            #
            try:
                concordance_cox_latent = nnet_model.coxfitter.score(test_cox_df, scoring_method="concordance_index")
            except:
                concordance_cox_latent = 0.0
            print('Concordance index (cox): ' + str(concordance_cox_latent))

            # if len(nnet_model.pred_time):
            #     pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy(),
            #                              concordance_indx.mean()[0], concordance_cox_latent])
            # else:
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy(),
                                     concordance_indx, concordance_cox_latent])

            if is_early_stopping:
                save_file = 'result_error_best.csv'
                torch.save(mu, os.path.join(results_path, 'z_recon_test_best.pth'))
            else:
                save_file = 'result_error.csv'
                torch.save(mu, os.path.join(results_path, 'z_recon_test.pth'))
            np.savetxt(os.path.join(results_path, save_file), pred_results)

    return pred_results


def test_CIndex_of_nnetmodel(dataset, nnet_model, num_workers=0):
    """
    Function to compute Mean Squared Error of test set with GP approximationö

    """

    print("Running tests with a test set")
    # dataset_type = 'HealthMNIST'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Length of dataset to be testes:  {}'.format(len(dataset)))
    dataloader_test = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=num_workers)
    concordance_indx = 0.0

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            surv_x = sample_batched['surv_covariates']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask'].to(device)
            data = data * mask.reshape(data.shape)

            recon_batch, mu, log_var = nnet_model(data)
            concordance_indx = calculate_risk_score(nnet_model, mu, label, surv_x, save_path=os.path.join(save_path,'test_results_2/'))

    return concordance_indx


def VAEtest(test_dataset, nnet_model, type_nnet, id_covariate):
    """
    Function to compute Mean Squared Error using just a VAE.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label'].to(device)
            data = sample_batched['digit'].to(device)
            mask = sample_batched['mask'].to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))


def HMNIST_test(nnet_model, type_nnet, latent_dim, prediction_x, prediction_mu, covar_module0, covar_module1, 
                likelihoods, csv_file_test_data, csv_file_test_label, test_mask_file, zt_list, P, T, id_covariate, 
                varying_T=False):
    """
    Function to compute Mean Squared Error of test set.
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file,
                                          root_dir='./data',
                                          transform=transforms.ToTensor())
    
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
