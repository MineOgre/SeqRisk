import argparse
import ast


"""
Helper code for loading parameters from parameter file or from command line
"""

class LoadFromFile (argparse.Action):
    """
    Read parameters from config file
    """
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().splitlines(), namespace)


class ModelArgs:
    """
    Runtime parameters for the L-VAE model
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the model')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=str, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_test_data', type=str, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_label', type=str, help='Name of label file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=str, help='Name of test label file', required=False)
        self.parser.add_argument('--csv_file_prediction_data', type=str, help='Name of prediction data file', required=False)
        self.parser.add_argument('--csv_file_prediction_label', type=str, help='Name of prediction label file', required=False)
        self.parser.add_argument('--csv_file_validation_data', type=str, help='Name of validation data file', required=False)
        self.parser.add_argument('--csv_file_validation_label', type=str, help='Name of validation label file', required=False)
        self.parser.add_argument('--csv_file_generation_data', type=str, help='Name of data file for image generation', required=False)
        self.parser.add_argument('--csv_file_generation_label', type=str, help='Name of label file for image generation', required=False)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--test_mask_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--prediction_mask_file', type=str, help='Name of prediction mask file', default=None)
        self.parser.add_argument('--validation_mask_file', type=str, help='Name of validation mask file', default=None)
        self.parser.add_argument('--generation_mask_file', type=str, help='Name of mask file for image generation', default=None)
        self.parser.add_argument('--dataset_type', required=False, choices=['RotatedMNIST', 'HealthMNIST', 'Physionet','HUSCorogene'],
                                 help='Type of dataset being used.')
        self.parser.add_argument('--latent_dim', type=int, default=2, help='Number of latent dimensions')
        self.parser.add_argument('--hidden_dim', type=int, default=64,
                                 help='Number of hidden dimensions for RNN')
        self.parser.add_argument('--risk_nn_conf', type=str2list, default='[]',
                                 help='NN Configuration of hazard function after input layer')
        self.parser.add_argument('--eval_time', type=str2list, default=[], help='Evaluation time for survival')
        self.parser.add_argument('--pred_time', type=str2list, default=[], help='Prediction time for survival')
        # self.parser.add_argument('--vae_nn_conf', type=str, default='[]',
        #                          help='NN Configuration of encoder and decoder(reverse of encoder) after input layer')
        self.parser.add_argument('--vae_nn_enc', type=str2list, default='[]',
                                 help='NN Configuration of encoder and decoder(reverse of encoder) after input layer')
        self.parser.add_argument('--vae_nn_dec', type=str2list, default='[]',
                                 help='NN Configuration of encoder and decoder(reverse of encoder) after input layer')
        self.parser.add_argument('--x_order', type=str2list, default='[]', help='Order of covariates for label')
        self.parser.add_argument('--surv_x', type=str2list, default='[]', help='Covariates to add the latent for survival')
        self.parser.add_argument('--time_bins', type=str2list, default='[0,0,0]', help='Min time, Max time, Bin Size')
        self.parser.add_argument('--d_model', type=int, default='64', help='Transformer model dimension')
        self.parser.add_argument('--id_covariate', type=int, help='Index of ID (unique identifier) covariate')
        self.parser.add_argument('--time_covariate', type=int, help='Index of stopping time covariate')
        self.parser.add_argument('--start_time_covariate', type=int, help='Index of start time covariate')
        self.parser.add_argument('--event_covariate', type=int, help='Index of event covariate')
        self.parser.add_argument('--risk_type', type=str, help='How to calculate the risk. cox_last_N ')
        self.parser.add_argument('--model_type', type=str, default='LVAE', help='Survival or not so far, LVAE or LVAE_survival')
        self.parser.add_argument('--M', type=int, help='Number of inducing points')
        self.parser.add_argument('--P', type=int, help='Number of unique instances')
        self.parser.add_argument('--T', type=int, help='Number of longitudinal samples per instance')
        self.parser.add_argument('--varying_T', type=str2bool, default=False, help='Varying number of samples per instance')
        self.parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
        self.parser.add_argument('--weight', type=float, default=1,
                                 help='Trade-off parameter balancing data reconstruction and latent space prior' +
                                      ' regularisation')
        self.parser.add_argument('--num_dim', type=int, help='Number of input dimensions', required=False)
        self.parser.add_argument('--num_samples', type=int, default=1, help='Number of Monte Carlo samples')
        self.parser.add_argument('--loss_function', type=str, default='mse', help='LVAE loss function for training (mse/nll)')
        self.parser.add_argument('--type_nnet', required=False, choices=['rnn', 'conv', 'simple'],
                                 help='Type of neural network for the encoder and decoder')
        self.parser.add_argument('--type_rnn', required=False, choices=['lstm', 'gru'],
                                 help='Type of rnn for rnn-encoder')
        self.parser.add_argument('--type_KL', required=False, choices=['closed', 'other', 'GPapprox', 'GPapprox_closed'],
                                 help='Type of loss computation')
        self.parser.add_argument('--constrain_scales', type=str2bool, default=False, required=False,
                                 help='Constrain the marginal variances')
        self.parser.add_argument('--model_params', type=str, default='model_params.pth',
                                 help='Pre-trained VAE parameters')
        self.parser.add_argument('--gp_model_folder', type=str, default='./pretrainedVAE',
                                 help='Pre-trained GP model parameters')
        self.parser.add_argument('--generate_plots', type=str2bool, default=False, help='Generate plots')
        self.parser.add_argument('--iter_num', type=int, default=1, help='Iteration number. Useful for multiple runs.')
        self.parser.add_argument('--test_freq', type=int, default=50, help='Period of computing test MSE.')
        self.parser.add_argument('--cat_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_kernel', type=ast.literal_eval)
        self.parser.add_argument('--sqexp_kernel', type=ast.literal_eval)
        self.parser.add_argument('--cat_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--covariate_missing_val', type=ast.literal_eval)
        self.parser.add_argument('--run_tests', type=str2bool, default=False,
                                 help='Perform tests using the trained model')
        self.parser.add_argument('--run_validation', type=str2bool, default=False,
                                 help='Test the model using a validation set')
        self.parser.add_argument('--generate_images', type=str2bool, default=False,
                                 help='Generate images of unseen individuals')
        self.parser.add_argument('--results_path', type=str, required=False, help='Path to results')
        self.parser.add_argument('--f', type=open, action=LoadFromFile)
        self.parser.add_argument('--mini_batch', type=str2bool, default=False, help='Use mini-batching for training.')
        self.parser.add_argument('--hensman', type=str2bool, default=False, help='Use true mini-batch training.')
        self.parser.add_argument('--variational_inference_training', type=str2bool, default=False, help='Use variational inference training.')
        self.parser.add_argument('--memory_dbg', type=str2bool, default=False, help='Debug memory usage in training')
        self.parser.add_argument('--natural_gradient', type=str2bool, default=True, help='Use natural gradients for parameters m and H')
        self.parser.add_argument('--natural_gradient_lr', type=float, default=0.01, help='Learning rate for variational parameters m and H if natural gradient is used')
        self.parser.add_argument('--subjects_per_batch', type=int, default=20, help='Number of subjects per batch in mini-batching.')
        self.parser.add_argument('--vy_fixed', type=str2bool, default=False, help='Use fixed variance for y in VAE')
        self.parser.add_argument('--vy_init', type=float, default=1.0, help='Initial variance for y in VAE')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Probability for dropout')
        self.parser.add_argument('--dropout_input', type=float, default=0., help='Probability for dropout at input layer')
        self.parser.add_argument('--is_early_stopping', type=str2bool, default=False, help='If we use early stopping model')
        self.parser.add_argument('--validation_criteria', type=str, default='mse', help='')
        self.parser.add_argument('--num_workers', type=int, default=0, help='Number of Workers for Dataloader')
        self.parser.add_argument('--cut_time', type=int, default=0, help='The Amount of time to cut from the end of the time series')
        self.parser.add_argument('--sigma', type=float, default=0.1, help='Divider in exponential funstion in ranking loss')
        self.parser.add_argument('--regression_ratio', type=float, required=False, help='Weight for regression loss')
        self.parser.add_argument('--beta', type=float, default=0.0, help='Multiplication factor for ranking loss in survival loss')
        self.parser.add_argument('--num_head', type=int, default=2, help='Number of Transformer Heads')
        self.parser.add_argument('--num_transformer_layer', type=int, default=2, help='Number of Transformer Encoder Layers')
        self.parser.add_argument('--dim_feedforward', type=int, default=2, help='Number of Dim Forward in Transformer')

    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt


class VAEArgs:
    """
    Runtime parameters for the VAE model
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the VAE pre-training')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=str, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_label', type=str, help='Name of label file', required=False)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--true_mask_file', type=str, help='Name of true mask file', default='')
        self.parser.add_argument('--true_test_mask_file', type=str, help='Name of true mask file for test data', default='')
        self.parser.add_argument('--true_prediction_mask_file', type=str, help='Name of true mask file for prediction data', default='')
        self.parser.add_argument('--true_validation_mask_file', type=str, help='Name of true mask file for validation data', default='')
        self.parser.add_argument('--csv_types_file', type=str, help='Name of types file', required=False)
        self.parser.add_argument('--csv_file_test_data', type=str, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=str, help='Name of test label file', required=False)
        self.parser.add_argument('--test_mask_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--csv_range_file', type=str, help='Name of types file', required=False)
        self.parser.add_argument('--csv_file_validation_data', type=str, help='Name of validation data file', required=False)
        self.parser.add_argument('--csv_file_validation_label', type=str, help='Name of validation label file', required=False)
        self.parser.add_argument('--validation_mask_file', type=str, help='Name of validation mask file', default=None)
        self.parser.add_argument('--dataset_type', required=False, choices=['RotatedMNIST', 'HealthMNIST', 'Physionet', 'Physionet2019', 'HeteroHealthMNIST', 'PPMI','HUSCorogene'],
                                 help='Type of dataset being used.')
        self.parser.add_argument('--batch_size', type=int, help='Batch size', default=500, required=False)
        self.parser.add_argument('--latent_dim', type=int, default=2, help='Number of latent dimensions')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for RNN')
        self.parser.add_argument('--hidden_layers', type=str, help='Number of hidden dimensions for hidden layers')
        self.parser.add_argument('--id_covariate', type=int, default=0, help='Index of ID (unique identifier) covariate')
        self.parser.add_argument('--time_covariate', type=int, default=0, help='Index of time covariate')
        self.parser.add_argument('--start_time_covariate', type=int, help='Index of start time covariate')
        self.parser.add_argument('--event_covariate', type=int, default=0, help='Index of event covariate')
        self.parser.add_argument('--y_dim', type=int, help='Number of Y dimensions', required=False)
        self.parser.add_argument('--s_dim', type=int, default=0, help='Number of S dimensions', required=False)
        self.parser.add_argument('--T', type=int, help='Number of longitudinal samples per instance')
        self.parser.add_argument('--varying_T', type=str2bool, default=False, help='Varying number of samples per instance')
        self.parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
        # self.parser.add_argument('--output_interval', type=int, default=10, help='Number of epochs for test outputs')
        self.parser.add_argument('--save_interval', type=int, default=10, help='Number of epochs for test outputs')
        self.parser.add_argument('--num_dim', type=int, help='Number of input dimensions', required=False)
        self.parser.add_argument('--type_nnet', required=False, choices=['rnn', 'conv', 'simple', 'hivae','conv-hivae'],
                                 help='Type of neural network for the encoder and decoder')
        self.parser.add_argument('--eval_time', type=str2list, default=[], help='Evaluation time for survival')
        self.parser.add_argument('--pred_time', type=str2list, default=[], help='Prediction time for survival')
        self.parser.add_argument('--conv_hivae', type=str2bool, default=False, help='Convolutional HIVAE')
        self.parser.add_argument('--conv_range', type=int, default=255, help='Convolutional Range')
        self.parser.add_argument('--use_ranges', type=str2bool, default=False, help='Indicator for Beta Likelihood')
        self.parser.add_argument('--type_rnn', required=False, choices=['lstm', 'gru'],
                                 help='Type of rnn for rnn-encoder')
        self.parser.add_argument('--loss_function', type=str, default='nll', help='VAE loss function for training (mse/nll)')
        self.parser.add_argument('--iter_num', type=int, default=1, help='Iteration number. Useful for multiple runs')
        self.parser.add_argument('--vy_fixed', type=str2bool, default=False, help='Use fixed variance for y in VAE')
        self.parser.add_argument('--vy_init', type=float, default=1.0, help='Initial variance for y in VAE')
        self.parser.add_argument('--vy_init_real', type=float, default=1.0, help='Initial variance for real y in HIVAE')
        self.parser.add_argument('--vy_init_pos', type=float, default=.5, help='Initial variance for positive y in HIVAE')
        self.parser.add_argument('--logvar_network', type=str2bool, default=False, help='Observation variance for real and positive dimensions are trained in HIVAE')
        self.parser.add_argument('--is_early_stopping', type=str2bool, default=False, help='Early Stopping')
        self.parser.add_argument('--run_tests', type=str2bool, default=False,
                                 help='Perform tests using the trained model')
        self.parser.add_argument('--model_params', type=str, default='model_params.pth',
                                 help='Pre-trained VAE parameters')
        self.parser.add_argument('--f', type=open, action=LoadFromFile)
        self.parser.add_argument('--risk_nn_conf', type=str2list, default='[]',
                                 help='NN Configuration of hazard function after input layer')
        # self.parser.add_argument('--vae_nn_conf', type=str, default='[]',
        #                          help='NN Configuration of encoder and decoder(reverse of encoder) after input layer')
        self.parser.add_argument('--vae_nn_enc', type=str2list, default='[]',
                                 help='NN Configuration of encoder and decoder(reverse of encoder) after input layer')
        self.parser.add_argument('--vae_nn_dec', type=str2list, default='[]',
                                 help='NN Configuration of encoder and decoder(reverse of encoder) after input layer')
        self.parser.add_argument('--x_order', type=str2list, default='[]', help='Order of covariates for label')
        self.parser.add_argument('--surv_x', type=str2list, default='[]', help='Covariates to add the latent for survival')
        self.parser.add_argument('--time_bins', type=str2list, default='[0,0,0]', help='Min time, Max time, Bin Size')
        self.parser.add_argument('--d_model', type=int, default='64', help='Transformer model dimension')
        self.parser.add_argument('--risk_type', type=str, help='How to calculate the risk. cox_last_N ')
        self.parser.add_argument('--model_type', type=str, default='VAE', help='Survival or not so far, VAE or VAE_survival')
        self.parser.add_argument('--validation_criteria', type=str, default='mse', help='')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Probability for dropout')
        self.parser.add_argument('--dropout_input', type=float, default=0., help='Probability for dropout at input layer')
        self.parser.add_argument('--num_workers', type=int, default=0, help='Number of Workers for Dataloader')
        self.parser.add_argument('--cut_time', type=int, default=0, help='The Amount of time to cut from the end of the time series')
        self.parser.add_argument('--sigma', type=float, required=False, default=0.1, help='Divider in exponential funstion in ranking loss')
        self.parser.add_argument('--regression_ratio', type=float, required=False, help='Weight for regression loss')
        self.parser.add_argument('--beta', type=float, required=False, default=0.0, help='Multiplication factor for ranking loss in survival loss')
        self.parser.add_argument('--num_head', type=int, default=2, help='Number of Transformer Heads')
        self.parser.add_argument('--num_transformer_layer', type=int, default=2, help='Number of Transformer Encoder Layers')
        self.parser.add_argument('--dim_feedforward', type=int, default=2, help='Number of Dim Forward in Transformer')

    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(l):
    return ast.literal_eval(l)
