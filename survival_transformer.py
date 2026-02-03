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
from VAE import SimpleVAE, ConvVAE, AttentionLayer, f_get_fc_mask1, f_get_fc_mask2

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
class TransSurvVAE(nn.Module):
    def __init__(self, input_dim, risk_nn_conf, risk_type, id_covariate, start_time_covariate,
                 time_covariate, event_covariate, model_type, beta, sigma, regression_ratio, surv_x_dim=0, dropout=0.,
                 time_bins=[0,0,0], d_model=32, num_head=2, num_transformer_layer=2, dim_feedforward=512, type_nnet='simple',
                 p = 0.5, latent_dim=20,
                 pred_time=None, eval_time=None):
        ## input_dim: dimension of the input data
        ## risk_nn_conf: configuration of the risk network
        ## risk_type: type of the risk network

        super(TransSurvVAE, self).__init__()
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
        self.type_nnet = type_nnet
        self.latent_dim = latent_dim
        self.p = p
        input_dim = input_dim + surv_x_dim

        if self.type_nnet == 'conv':
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
            input_dim = self.latent_dim

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
                layer_conf = [d_model*self.num_Category] + risk_nn_conf
            else:
                layer_conf = [d_model] + risk_nn_conf
        else:
            layer_conf = [input_dim*self.last_N] + risk_nn_conf


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
            self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_transformer_layer)

            # self.monitored_transformer_encoder = build_monitored_transformer_encoder(1, d_model, 2, d_model,
            #                                                                     dropout)

            # self.monitored_transformer_encoder = transformer_encoder(1, d_model, 2, d_model,
            #                                                                     dropout)

        # Initialize the attention layer
            if 'long' not in model_type:
                self.attention_layer = AttentionLayer(feature_dim=d_model)  # d_model is the dimension of the encoder outputs

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
            self.hazard_net.append(nn.Linear(layer_conf[0], layer_conf[1] , bias=False))
        self.hazard_net = nn.Sequential(*self.hazard_net)

        # hook = self.hazard_net.register_backward_hook(backward_hook)

    def forward_conv(self, x):
        z = F.relu(self.conv1(x))
        z = self.dropout2d_1(self.pool1(z))
        z = F.relu(self.conv2(z))
        z = self.dropout2d_2(self.pool2(z))

        # MLP
        z = z.view(-1, 32 * 9 * 9)
        h1 = self.dropout1(F.relu(self.fc1(z)))
        h2 = self.dropout2(F.relu(self.fc21(h1)))
        return self.fc211(h2)

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
            output = output.transpose(0,1).reshape(-1, self.num_Category * self.d_model)

        # if last_meas not in [None, []]:
        #     output = torch.cat((output, last_meas), dim=1)

        return self.hazard_net(output)

    def risk_loss_transformer(self, structured_mu, E, time_data, last_meas_time, last_maes, src_mask):
        # Shared functionality for both ConvVAE and SimpleVAE
        # print("This is a shared method.")

        # # Project the features to match d_model
        # risk = self.risk_forward(structured_mu,src_mask.bool())
        risk = self.risk_forward(structured_mu,src_mask.bool(), last_maes)

        sort_values = time_data * 10 ** 4 + E

        # Get the sorted indices
        sorted_indices = torch.argsort(sort_values, descending=True)

        # Using these indices to rearrange the risk and events tensors
        sorted_risk = risk[sorted_indices]
        sorted_E = E[sorted_indices]

        # Now you can use the sorted risk and events tensors in your loss function
        surv_nll = loss_function_risk_reg(sorted_risk, sorted_E)

        return surv_nll, risk

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
                indices = (train_x[:,self.id_covariate] == subject_id).nonzero(as_tuple=True)[0]
                times = train_x[:,self.start_time_covariate][indices]
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
        nnet_model = TransSurvVAE(num_dim, risk_nn_conf, 'last_1', id_covariate, start_time_covariate, time_covariate, event_covariate,
                surv_x_dim=len(surv_x),
                model_type=model_type, beta=beta, sigma=sigma, regression_ratio=regression_ratio,
                 time_bins=time_bins, type_nnet=type_nnet, dropout=dropout, latent_dim=latent_dim,
                d_model=d_model,
                num_head=num_head, dim_feedforward=dim_feedforward, num_transformer_layer=num_transformer_layer,
                pred_time=pred_time, eval_time=eval_time).double().to(device)

    elif type_nnet == 'simple':
        print('Using standard MLP')
        # nnet_model = SimpleVAE(latent_dim, num_dim, vy, vy_fixed).to(device)
        nnet_model = TransSurvVAE(num_dim, risk_nn_conf, 'last_1', id_covariate, start_time_covariate, time_covariate,
                               event_covariate, model_type=model_type, beta=beta, sigma=sigma, regression_ratio=regression_ratio, dropout=0.,
                                    d_model=d_model, num_head=num_head, num_transformer_layer=2, surv_x_dim=len(surv_x),
                                  dim_feedforward=dim_feedforward, time_bins=time_bins).double().to(device)


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
        if is_early_stopping:
            break
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
            masked_data = torch.mul(data, mask.reshape(data.shape))
            label = sample_batched['label'].double().to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate + 1:]), dim=1)
            surv_covs = sample_batched['surv_covariates'].double().to(device)

            optimiser.zero_grad()  # clear gradients

            # recon_batch, mu, log_var = nnet_model(data)
            #
            # [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)


            if nnet_model.model_type == 'VAE':
                survival_loss = torch.tensor(0.0, dtype=torch.double).to(device)
            elif 'survival' in nnet_model.model_type:
                if nnet_model.transformer:
                    if nnet_model.type_nnet == 'conv':
                        masked_data = nnet_model.forward_conv(masked_data)
                    last_meas, structured_mu, E, time_data, src_padding_mask, last_meas_time = nnet_model.get_survival_data(
                        masked_data,
                        label, surv_covs)

                    survival_loss, out = nnet_model.risk_loss_transformer(structured_mu.double(), E, time_data,
                                                                          last_meas_time, last_meas,
                                                                          src_padding_mask.double())
                else:
                ### Here the survival loss is the average of observed events
                    survival_loss = nnet_model.risk_loss(nnet_model.sample_latent(mu,log_var), label, surv_covs)
                # Assuming 'tensor' is the tensor you want to track

            # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            if loss_function == 'nll':
                survival_loss_sum += survival_loss.item()
                # net_loss = (nll_loss + kld_loss) * (1 - regression_ratio) + regression_ratio * survival_loss
                # Calculate the two parts of the loss

                loss = survival_loss

            if torch.isnan(loss).any():
                print("net_loss contains NaN values!")
                survival_loss = nnet_model.risk_loss(nnet_model.sample_latent(mu,log_var), label, surv_covs)

            loss.backward()  # compute gradients
            train_loss += loss.item()
            # recon_loss_sum += recon_loss.sum().item()
            # nll_loss += nll.sum().item()
            # kld_loss += KLD.sum().item()

            optimiser.step()  # update parameters

        print(
            '====> Epoch: {}/{} - Duration: {:.4f}  - Average loss: {:.4f}  - KLD loss: {:.3f}  - NLL loss: {:.3f}  - Recon loss: {:.3f} - Survival Loss: {:.4f}'.format(
                epoch, epochs, time.time()-start_time, train_loss, kld_loss, nll_loss, recon_loss_sum,survival_loss_sum))
        net_train_loss = np.append(net_train_loss, train_loss)

        # if epoch > 1:
        #     # log metrics to wandb
        #     wandb.log({"net_loss_sum": train_loss,
        #                "recon_loss_sum": recon_loss_sum, "nll_loss_sum": nll_loss,
        #                "kld_loss_sum": kld_loss,
        #                "survival_loss_sum": survival_loss_sum })

        if (csv_file_validation_data and not (epoch % validation_interval)) or is_early_stopping: # and epoch > 100:
            print("Testing the model with a validation set")
            batch_size_val = len(validation_dataset)
            subjects_per_batch = validation_dataset.label_source['subject'].nunique()
            validation_dataloader = HensmanDataLoader(validation_dataset, batch_sampler=VaryingLengthBatchSampler(
                VaryingLengthSubjectSampler(validation_dataset, nnet_model.id_covariate), subjects_per_batch),
                                                      num_workers=num_workers)
            validation_loss = torch.tensor(0.).to(device)
            validation_recon_loss = torch.tensor(0.).to(device)
            if nnet_model.type_nnet == 'conv':
                full_data = torch.zeros(len(validation_dataset), nnet_model.latent_dim, dtype=next(nnet_model.parameters()).dtype).to(device)
            else:
                full_data = torch.zeros(len(validation_dataset),validation_dataset.data_source.shape[1], dtype=next(nnet_model.parameters()).dtype).to(device)
            full_label = torch.zeros(len(validation_dataset), label.shape[1], dtype=next(nnet_model.parameters()).dtype).to(device)
            full_surv_covs = torch.zeros(len(validation_dataset), len(surv_x), dtype=next(nnet_model.parameters()).dtype).to(device)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    indices = sample_batched['idx']
                    data = sample_batched['digit'].double().to(device)
                    mask = sample_batched['mask'].to(device)
                    masked_data = torch.mul(data, mask.reshape(data.shape))
                    label = sample_batched['label'].double().to(device)
                    surv_covs = sample_batched['surv_covariates'].double().to(device)
                    # recon_batch, mu_qz, log_var_qz = nnet_model(masked_data)
                    # [recon_loss, nll] = nnet_model.loss_function(recon_batch, masked_data, mask)
                    if nnet_model.type_nnet == 'conv':
                        masked_data = nnet_model.forward_conv(masked_data)
                    full_data[indices] = masked_data
                    full_label[indices] = label
                    full_surv_covs[indices] = surv_covs

                    # KLD = -0.5 * torch.sum(1 + log_var_qz - mu_qz.pow(2) - log_var_qz.exp(), dim=1)
                    if nnet_model.model_type == 'VAE':
                        survival_loss = torch.tensor(0.0, dtype=torch.double).to(device)
                    elif 'survival_transformer' in nnet_model.model_type:
                        last_meas, structured_mu, E, time_data, src_padding_mask, last_meas_time = nnet_model.get_survival_data(
                            masked_data,
                            label, surv_covs)

                        survival_loss, out = nnet_model.risk_loss_transformer(structured_mu.double(), E, time_data,
                                                                              last_meas_time, last_meas,
                                                                              src_padding_mask.double())

                    validation_loss += survival_loss
                    # validation_recon_loss += recon_loss.sum().item()
            print("Calculating C-index for validation set")
            try:
                print("Line 573")
                c_index_val = calculate_risk_score(nnet_model, full_data, full_label, full_surv_covs,
                                                   save_path=os.path.join(save_path,'validation_results'))
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
            # if is_early_stopping:
            #     torch.save(full_mu, os.path.join(save_path, 'z_valid_best.pth'))
            # else:
            #     torch.save(full_mu, os.path.join(save_path, 'z_valid.pth'))
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
                    np.savetxt(os.path.join(save_path, 'best_epoch.csv'), np.array(epoch).reshape([1,1]))
                    print('Best model saved at epoch {}'.format(epoch))
                    # Stack the results
                    pred_results = np.array(
                        [validation_loss.detach().cpu().numpy(), validation_recon_loss.detach().cpu().numpy(),
                         c_index_val])
                    df_results = pd.DataFrame(pred_results[:, None], index=["NLL", "MSE", "C-Index"])

                    # Save to CSV
                    df_results.to_csv(os.path.join(save_path, "best_validation_scores.csv"), index=True,
                                      header=False)

        if epoch %  5== 0:
            # print(nnet_model.vy)
            if run_tests:
                test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
                                        num_workers=num_workers)
                for batch_idx, sample_batched in enumerate(test_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    masked_data = torch.mul(data, mask.reshape(data.shape))
                    label = sample_batched['label'].double().to(device)
                    surv_covs = sample_batched['surv_covariates'].double().to(device)
                    if nnet_model.type_nnet == 'conv':
                        masked_data = nnet_model.forward_conv(masked_data)
                print("Line 663")
                c_index_test = calculate_risk_score(nnet_model, masked_data, label, surv_covs,
                                                   save_path=os.path.join(save_path, f'test_results{epoch}'))

    if run_tests and is_early_stopping:
        # VAEtest(test_dataset, nnet_model, type_nnet, id_covariate)
        dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)
        for batch_idx, sample_batched in enumerate(dataloader):
            data = sample_batched['digit'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            masked_data = torch.mul(data, mask.reshape(data.shape))
            if nnet_model.type_nnet == 'conv':
                masked_data = nnet_model.forward_conv(masked_data)
            label = sample_batched['label'].double().to(device)
            surv_covs = sample_batched['surv_covariates'].double().to(device)
        print("Line 646")
        c_index_best_test = calculate_risk_score(nnet_model, masked_data, label, surv_covs,
                                           save_path=os.path.join(save_path, 'best_test_results'))
        pred_results = np.array(
            [0, 0,
             c_index_best_test])
        df_results = pd.DataFrame(pred_results[:, None], index=["NLL", "MSE", "C-Index"])

        # Save to CSV
        df_results.to_csv(os.path.join(save_path, "best_test_results.csv"), index=True,
                          header=False)

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
                if nnet_model.type_nnet == 'conv':
                    masked_data = nnet_model.forward_conv(masked_data)
                label = sample_batched['label'].to(device).double()
                surv_covs = sample_batched['surv_covariates'].double().to(device)


        print("Calculating C-index for validation set")
        try:
            print("Line 641")
            c_index_test = calculate_risk_score(nnet_model, masked_data, label, surv_covs,
                                                save_path=os.path.join(save_path, f'best_valiation_results'))
            pred_results = np.array(
                [0, 0,
                 c_index_test])
            df_results = pd.DataFrame(pred_results[:, None], index=["NLL", "MSE", "C-Index"])

            # Save to CSV
            df_results.to_csv(os.path.join(save_path, "best_valiation_results.csv"), index=True,
                              header=False)
        except:
            c_index_val = 0.
            print("Error calculating C-index for validation set")

    # print(nnet_model.vy)
    if not is_early_stopping:
        torch.save(nnet_model.state_dict(), os.path.join(save_path, 'model_params_vae.pth'))
