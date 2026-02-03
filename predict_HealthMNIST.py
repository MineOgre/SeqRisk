import os
import sys

import pandas as pd
from lifelines.utils import concordance_index


sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset_def import HealthMNISTDatasetConv, HealthMNISTDataset
from utils import batch_predict, batch_predict_varying_T, cox_loss_last_n, calculate_risk_score, HensmanDataLoader, \
    VaryingLengthBatchSampler, VaryingLengthSubjectSampler


def gen_rotated_mnist_plot(X, recon_X, labels, seq_length=16, num_sets=3, save_file='recon.pdf'):
    """
    Function to generate rotated MNIST digits plots.
    
    """
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    fig.set_size_inches(9, 1.5 * num_sets)
    for j in range(num_sets):
        begin = seq_length * j
        end = seq_length * (j + 1)
        time_steps = labels[begin:end, 1]
        for i, t in enumerate(range(end-begin)):
            try:
                ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray')
                ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray')
            except:
                print('Error in plotting Line 43')
    plt.savefig(save_file)
    plt.close('all')

def gen_rotated_mnist_seqrecon_plot_old(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate rotated MNIST digits.
    
    """
    num_sets = 4
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(3 * num_sets, 3 * num_sets)

    for j in range(num_sets):
        begin = seq_length_train * j
        end = seq_length_train * (j + 1)
        time_steps = labels_train[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray')

        begin = seq_length_full * j
        end = seq_length_full * (j + 1)
        time_steps = labels_recon[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file)
    plt.close('all')


def gen_rotated_mnist_seqrecon_plot_varying_T(Y, recon_Y, pred_Y, labels_recon, labels_train,
                                              id_covariate,
                                              time_covariate,
                                              save_file='recon_complete.pdf'):
    """
    Function to generate Health MNIST digits.

    """
    num_sets = 8
    fig, ax = plt.subplots(3 * num_sets + 1, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
            ax__.axis('off')
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(12, 20)
    # time_covariate = 1

    subject_ids = labels_train[:, id_covariate].unique()
    # Generate a permutation of indices
    perm = torch.randperm(subject_ids.size(0))
    # Shuffle the tensor
    subject_ids = subject_ids[perm]

    for j in range(num_sets):
        # get the jth subject id
        subject_id = subject_ids[j]
        # get the indices of the subject id
        indices = np.where(labels_train[:, id_covariate] == subject_id)[0]
        # get the time steps of the subject id
        time_steps = labels_train[indices, time_covariate]
        # set the time steps in order and assign sequential indices starting from 0
        time_steps = np.sort(time_steps)
        time_steps = np.array([np.where(time_steps == t)[0][0] for t in time_steps])
        # get the data corresponding to the subject id
        Y_subject = Y[indices, :]
        recon_Y_subject = recon_Y[indices, :]
        pred_Y_subject = pred_Y[indices, :]

        # begin_data = seq_length_train * j
        # end_data = seq_length_train * (j + 1)

        # begin_label = seq_length_full*2*j
        # mid_label = seq_length_full*(2*j+1)
        # end_label = seq_length_full*2*(j+1)

        # time_steps = labels_train[begin_data:end_data, time_covariate]
        for i, t in enumerate(time_steps):
            try:
                ax[3 * j, int(t)].imshow(np.reshape(Y_subject[i,:], [36, 36]), cmap='gray')
            except:
                print('Error in plotting Line 133')

        # time_steps = labels_train[begin_data:end_data, time_covariate]
        for i, t in enumerate(time_steps):
            ax[3 * j + 1, int(t)].imshow(np.reshape(recon_Y_subject[i,:], [36, 36]), cmap='gray')

        # time_steps = labels_train[begin_data + 160:end_data + 160, time_covariate]
        for i, t in enumerate(time_steps):
            ax[3 * j + 2, int(t)].imshow(np.reshape(pred_Y_subject[i,:], [36, 36]), cmap='gray')

    plt.savefig(save_file, bbox_inches='tight')
    plt.close('all')


def gen_rotated_mnist_seqrecon_plot(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate Health MNIST digits.
    
    """    
    num_sets = 4
    fig, ax = plt.subplots(4 * num_sets - 1, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
            ax__.axis('off')
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(12, 20)
    time_covariate = 1

    for j in range(num_sets):
        begin_data = seq_length_train*j
        end_data = seq_length_train*(j+1)

        # begin_label = seq_length_full*2*j
        # mid_label = seq_length_full*(2*j+1)
        # end_label = seq_length_full*2*(j+1)
        
        time_steps = labels_train[begin_data:end_data, time_covariate]
        for i, t in enumerate(np.arange(len(time_steps))):
            try:
                ax[4 * j, int(t)].imshow(np.reshape(X[begin_data + i, :], [36, 36]), cmap='gray')
            except:
                print('Error in plotting Line 178')
        
        time_steps = labels_train[begin_data:end_data,time_covariate]
        for i, t in enumerate(np.arange(len(time_steps))):
            ax[4 * j + 1, int(t)].imshow(np.reshape(recon_X[begin_data + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[begin_data+160:end_data+160, time_covariate]
        for i, t in enumerate(np.arange(len(time_steps))):
            ax[4 * j + 2, int(t)].imshow(np.reshape(recon_X[begin_data + i + 160, :], [36, 36]), cmap='gray')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close('all')

def recon_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                       covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, 
                       prediction_mu, epoch, zt_list, P, T, id_covariate, num_workers, varying_T=False):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask'].to(device)
            masked_data = data * mask.reshape(data.shape)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(masked_data)
            # Z = nnet_model.sample_latent(mu, log_var)

            # Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)

            filename = 'recon_complete.pdf' if epoch == -1 else 'recon_complete_best.pdf' if epoch == -2 else 'recon_complete_' + str(epoch) + '.pdf'

            # gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
            #                                 save_file=os.path.join(results_path, filename))
            if varying_T:
                gen_rotated_mnist_seqrecon_plot_varying_T(masked_data.cpu(), recon_batch.cpu(), recon_Z.cpu(), label.cpu(), label.cpu(),
                                                          nnet_model.id_covariate, nnet_model.time_covariate,
                                            save_file=os.path.join(results_path, filename))
            else:
                gen_rotated_mnist_seqrecon_plot(masked_data[0:160, :].cpu(), torch.cat([recon_batch[0:160, :].cpu(), recon_Z[0:160, :].cpu()]), label[0:320, :].cpu(),
                                            torch.cat([label[0:160, :].cpu(), label[0:160, :].cpu()]),
                                            save_file=os.path.join(results_path, filename))

def variational_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                             covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, 
                             prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Length of generation dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            print('Prediction size: ' + str(Z_pred.shape))
            recon_Z = nnet_model.decode(Z_pred)
            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, 'recon_complete_' + str(epoch) + '.pdf'))

def VAEoutput(nnet_model, dataset, epoch, save_path, type_nnet, id_covariate, num_workers, is_early_stopping=False):
    """
    Function to obtain output of VAE.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Batch size must be a multiple of T
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
            # no mini-batching. Instead get a mini-batch of size 4000
            label = sample_batched['label'].double().to(device)
            data = sample_batched['digit'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            # masked_data = torch.mul(data, mask.reshape(data.shape))
            surv_covariate = sample_batched['surv_covariates'].double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            if nnet_model.risk_type is not None:
                print("Calculating C-index for test set")
                concordance_indx = calculate_risk_score(nnet_model, mu, label, surv_covariate,
                                                        save_path=os.path.join(save_path, 'test_results'))
            else:
                concordance_indx = 0.

            if is_early_stopping:
                save_file = 'result_error_best.csv'
                torch.save(mu, os.path.join(save_path, 'z_recon_test_best.pth'))
            else:
                save_file = 'result_error.csv'
                torch.save(mu, os.path.join(save_path, 'z_recon_test.pth'))
            [recon_loss, _] = nnet_model.loss_function(recon_batch, data, mask)
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), 0.,
                                    concordance_indx])
            np.savetxt(os.path.join(save_path, save_file), pred_results)

            if is_early_stopping:
                gen_rotated_mnist_plot(data[40:200, :].cpu(), recon_batch[40:200, :].cpu(), label[40:200, :].cpu(), seq_length=20, num_sets=8,
                                       save_file=os.path.join(save_path, 'recon_VAE_best.pdf'))
            else:
                gen_rotated_mnist_plot(data[40:200, :].cpu(), recon_batch[40:200, :].cpu(), label[40:200, :].cpu(), seq_length=20, num_sets=8,
                                   save_file=os.path.join(save_path, 'recon_VAE_' + str(epoch) + '.pdf'))
            break

def predict_generate(csv_file_test_data, csv_file_test_label, csv_file_test_mask, dataset, generation_dataset, nnet_model, results_path, covar_module0, covar_module1, likelihoods, type_nnet, latent_dim, data_source_path, prediction_x, prediction_mu, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to perform prediction and visualise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    print('Length of dataset:  {}'.format(len(dataset)))

    # set up Data Loader
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers)

    # Get values for GP initialisation:
    Z = torch.tensor([]).to(device)
    mu = torch.tensor([]).to(device)
    log_var = torch.tensor([]).to(device)
    data_train = torch.tensor([]).to(device)
    label_train = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
            # no mini-batching. Instead get a mini-batch of size 4000
            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            label_train = label
            data_train = data
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            gen_rotated_mnist_plot(data[40:100, :].cpu(), recon_batch[40:100, :].cpu(), label[40:100, :].cpu(), seq_length=20,
                                   save_file=os.path.join(results_path, 'recon_train.pdf'))
            break

    if type_nnet == 'conv':
        test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                              csv_file_label=csv_file_test_label,
                                              mask_file=csv_file_test_mask, root_dir=data_source_path,
                                              transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=csv_file_test_mask, root_dir=data_source_path,
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
            
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            np.savetxt(os.path.join(results_path, 'result_error.csv'), pred_results)

    print('Length of generation dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            print('Prediction size: ' + str(Z_pred.shape))
            recon_Z = nnet_model.decode(Z_pred)

            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, 'recon_complete.pdf'))

