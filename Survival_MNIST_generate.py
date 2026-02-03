import os
import glob
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy import ndimage
import matplotlib.pyplot as plt
import argparse
import random


"""
Code to generate the Health MNIST data.

This code manipulates the original MNIST images as described in the L-VAE paper.
See: http://arxiv.org/abs/2006.09763
"""

def parse_arguments():
    """
    Parse the command line arguments
    :return: parsed arguments object (2 arguments)
    """

    parser = argparse.ArgumentParser(description='Enter configuration for generating data')
    parser.add_argument('--source', type=str, default='./trainingSet', help='Path to MNIST image root directory')
    parser.add_argument('--destination', type=str, default='./data', help='Path to save the generated dataset')
    parser.add_argument('--num_3', type=int, default=100, help='Number of unique instances for digit 3')
    parser.add_argument('--num_6', type=int, default=100, help='Number of unique instances for digit 6')
    parser.add_argument('--additive_noise', type=int, default=0, help='Additive Noise amount')
    parser.add_argument('--event_prob', type=float, default=0.5,
                        help='Event Probability among the population')
    parser.add_argument('--missing', type=float, default=25, #choices=range(-1, 101),
                        help='Percentage of missing in range [0, 100]')
    parser.add_argument('--image_folder_name', type=str, default='./data/images',
                        help='Folder name of generated images')
    parser.add_argument('--data_file_name', type=str, default='health_MNIST_data.csv',
                        help='File name of generated data')
    parser.add_argument('--data_masked_file_name', type=str, default='health_MNIST_data_masked.csv',
                        help='File name of generated masked data')
    parser.add_argument('--labels_file_name', type=str, default='health_MNIST_label.csv',
                        help='File name of generated labels')
    parser.add_argument('--mask_file_name', type=str, default='mask.csv',
                        help='File name of generated mask')
    parser.add_argument('--digit_type', type=str, default='one',
                        help='File name of generated mask')
    return vars(parser.parse_args())

min_im_num = 7
def biased_random(min_value=40, max_value=70, bias=2.0):
    """Generate a biased random float between min_value and max_value.

    The bias determines the likelihood of larger numbers:
    - bias > 1.0 makes larger numbers more likely.
    - bias < 1.0 makes smaller numbers more likely.
    - bias = 1.0 is unbiased.
    """
    return min_value + (max_value - min_value) * (random.random() ** (1.0 / bias))



def create_data_file(path, open_str):
    if os.path.exists(path):
        os.remove(path)
    return open(path, open_str)

def write_label_file_header(label_file):
    df = pd.DataFrame.from_dict({}, orient='index',
                                columns=['subject', 'age', 'event', 'gender', 'location', 'angle',
                                         'time',
                                         'time_age', 'digit'])
    df.to_csv(label_file, index=False)

def save_data(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, missing_frac, observed_frac):

    # save rotated MNIST
    np.savetxt(data_file, rotated_MNIST, fmt='%d', delimiter=',')
    
    # generate mask
    mask = np.random.choice([0, 1], size=rotated_MNIST.shape, p=[missing_frac, observed_frac])
    
    # 0 implies missing, 1 implies observed
    masked_data = np.multiply(rotated_MNIST, mask)

    np.savetxt(data_masked_file, masked_data, fmt='%d', delimiter=',')
    np.savetxt(mask_file, mask, fmt='%d', delimiter=',')

    df = pd.DataFrame.from_dict(label_dict, orient='index',
                                columns=['subject', 'digit', 'angle', 'event',
                                         'time', 'gender',
                                         'time_age', 'location'])

    # save labels
    df.to_csv(label_file, index=False, header=False)


def save_data_pd(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, missing_frac,
              observed_frac):
    # save rotated MNIST
    pd_rotatedMNIST = pd.DataFrame(rotated_MNIST)
    pd_rotatedMNIST.to_csv(data_file, mode='a', header=False, index=False)

    np.savetxt(data_file, rotated_MNIST, fmt='%d', delimiter=',')

    # generate mask
    mask = np.random.choice([0, 1], size=rotated_MNIST.shape, p=[missing_frac, observed_frac])

    # 0 implies missing, 1 implies observed
    masked_data = np.multiply(rotated_MNIST, mask)

    df_masked_data = pd.DataFrame(masked_data).astype(int)
    df_mask = pd.DataFrame(mask)
    df_masked_data.to_csv(data_masked_file, mode='a', header=False, index=False)
    df_mask.to_csv(mask_file, mode='a', header=False, index=False)

    df = pd.DataFrame.from_dict(label_dict, orient='index',
                                columns=['subject', 'start_obs', 'stop_obs', 'event', 'gender', 'location', 'start_age', 'stop_age', 'angle',
                                         'disease_time',
                                         'time_age', 'digit'])

    # save labels
    df.to_csv(label_file, mode='a', index=False, header=df.columns)


def generate_time_points(last_seen_time_point, num_of_observations, min_distance=0.1):
    time_points = np.sort(np.random.uniform(0, last_seen_time_point, num_of_observations))
    diffs = np.diff(time_points)
    max_iter = 20  # Maximum number of iterations
    start_deleting_point = max_iter - min_im_num

    for i in range(max_iter):
        # Identify points which are too close to the next one
        too_close = np.where(diffs < min_distance)[0]

        if not len(too_close):
            return time_points

        if i > start_deleting_point:
            # If we're here, we couldn't fix all points by resampling. Remove problematic points.
            print(f"Couldn't fix all points by resampling, removing 1 point")
            try:
                time_points = np.delete(time_points, too_close[0])
            except:
                print("Couldn't delete point")
            too_close = np.delete(too_close, 0)
            # diffs = np.diff(time_points)
            # continue

        # If there are no problematic points, we're done
        if not len(too_close):
            assert len(time_points) >= min_im_num, "Couldn't fix all points by resampling"
            return time_points

        # Resample only those points
        for index in too_close:
            time_points[index] = np.random.uniform(
                0,
                last_seen_time_point
            )

        # Re-sort and recompute differences
        time_points = np.sort(time_points)
        diffs = np.diff(time_points)

    # If we're here, we couldn't fix all points by resampling. Remove problematic points.
    print(f"Couldn't fix all points by resampling, removing {len(too_close)} points")
    time_points = np.delete(time_points, too_close)

    diffs = np.diff(time_points)
    too_close = np.where(diffs < min_distance)[0]

    if len(time_points) - len(too_close) < min_im_num:
        N = len(time_points) - min_im_num
        # Find the indices of the N smallest diffs
        smallest_diff_indices = np.argsort(diffs)[:N]
        # Remove the corresponding time points; note that we might need to adjust indices
        # because removing an element affects subsequent indices
        time_points = np.delete(time_points, smallest_diff_indices + 1)
    else:
        print(f"Couldn't fix all points by resampling again, removing {len(too_close)} points")
        time_points = np.delete(time_points, too_close)

    print(f"Remaining time points: {len(time_points)}")
    assert len(time_points) > 4, "Couldn't fix all points by resampling"

    return time_points


if __name__ == '__main__':
    opt = parse_arguments()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    digit_mod = {'3': num_3, '6': num_6}
    # event_prob = 0.5  # probability of instance being dead
    sample_index = 0
    subject_index = 0
    label_dict = {}
    gender = 0
    dead_number = 0
    # observation_time_points = 20
    # rotation_time_points = 40
    total_years = 6

    # 20 time points
    # time_age = np.arange(0, observation_time_points)
    # time_points = np.arange(1, rotation_time_points+1)

    # accumulate digits
    rotated_MNIST = np.empty((0, 1296))

    # if destination folder does not exist, create it with access everyone
    if not os.path.exists(destination):
        os.makedirs(destination)
        os.chmod(destination, 0o777)

    if not os.path.exists(image_folder_name):
        os.makedirs(image_folder_name)
        os.chmod(image_folder_name, 0o777)

    data_file = create_data_file(os.path.join(destination, data_file_name), "ab")
    mask_file = create_data_file(os.path.join(destination, mask_file_name), "ab")
    data_masked_file = create_data_file(os.path.join(destination, data_masked_file_name), "ab")
    label_file = create_data_file(os.path.join(destination, labels_file_name), "a")
    write_label_file_header(label_file) 

    missing_frac = missing/100
    observed_frac = 1 - missing_frac

    type = 'all'
    if type == 'test':
        subject_index = 1001

    for digit in digit_mod.keys():
        print("Creating instances of digit {}".format(digit))

        # read in the files
        data_path = os.path.join(source, digit)
        files = glob.glob('{}/*.jpg'.format(data_path))

        if type == 'test':
            files = sorted(files, reverse=True)
        else:
            files = sorted(files)
        # Assume requested files less than total available!
        np.random.seed(100)
        random.seed(50)
        images = []
        for i in range(digit_mod[digit]):
            ## Just one digit for the given number of subjects
            if digit_type == 'different':
                original_image = plt.imread(files[i])
            else:
                original_image = plt.imread(files[10])
            original_image_pad = np.pad(original_image, ((4, 4), (4, 4)), 'constant')

            # decide on sickness
            event_var = np.random.binomial(1, event_prob)
            if event_var:
                dead_number += 1

            # irrelevant location
            loc_var = np.random.binomial(1, 0.5)

            # introduce some noise
            num_of_observations = np.random.randint(min_im_num, 20)  # number of observations

            # Shape parameters
            alpha = 2  # controls shape in one direction
            beta = 10  # controls shape in the other direction

            # Generate random numbers
            ##### According to Beta Distribution
            # last_seen_time_point = 180 - np.random.beta(alpha, beta, 1) * 180
            ##### According to Uniform Distribution between mid way and last possible point
            last_seen_time_point = np.random.uniform(30, 180)
            # time_points = np.sort(np.random.uniform(0, last_seen_time_point, num_of_observations))
            time_points = generate_time_points(last_seen_time_point, num_of_observations)
            # rotations = np.random.normal(0, 2, len(time_points))
            start_age = biased_random()
            time_points_age = time_points/180 * total_years + start_age
                    # np.arange(0, rotation_time_points) * (
                    # ((total_years * 12)) / (rotation_time_points - 1)) / 12)

            # Angle per step
            angles = time_points
            # rotations += angles
            rotations = angles

            shift = np.random.uniform(-2, 2)
            # shift = 0
            # noise = np.random.randint(-additive_noise, additive_noise, (36,36))
            mean = 0
            std = additive_noise
            noise = np.random.normal(mean, std, (36,36))
            if i<20:
                start_point = 0
                rows = 1
                cols = int(np.ceil(len(rotations) / rows))

                # fig, axes = plt.subplots(nrows=len(rotations), figsize=(15, len(rotations)))

                for idx, rotation in enumerate(rotations):
                    # rotate an instance
                    img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)

                    # diagonal shift the image
                    img = ndimage.shift(img, shift=shift)
                    img = np.clip((img+noise).astype(int) , 0, 255).astype(np.uint8)

                    # rotated_MNIST = np.append(rotated_MNIST, np.reshape(img, (1, 1296)), axis=0)

                    plt.subplot(rows, cols, idx + 1)
                    plt.imshow(img * np.random.choice([0, 1], size=(36,36), p=[missing_frac, 1- missing_frac]))
                    # plt.imshow(img)
                    plt.title(f'{idx + start_point + 1}')
                    plt.axis('off')

                # plt.title(f'{digit} {start_point} {event_var}')
                plt.savefig(
                    os.path.join(image_folder_name, 'digit_{}_{}_{}_all_{}.pdf'.format(digit, subject_index, event_var,i)))
                plt.show()
                plt.close()
                images.append(img)

            # start_point = np.random.randint(1, rotation_time_points - observation_time_points - 1)
            # rotations = rotations[start_point:start_point + observation_time_points]


            if digit == '3':
                gender = 0
            else:
                gender = 1

            for idx, rotation in enumerate(rotations):

                # rotate an instance
                img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)

                # diagonal shift the image
                img = ndimage.shift(img, shift=shift)
                img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

                rotated_MNIST = np.append(rotated_MNIST, np.reshape(img, (1, 1296)), axis=0)

                start_point = rotations[idx]
                if idx+1 != len(rotations):
                    stop_obs = rotations[idx+1]
                    stop_age = time_points_age[idx+1]
                else:
                    stop_obs = 180 if event_var else np.random.uniform(start_point, 180-5)
                    stop_age = (total_years+time_points_age[0]) if event_var else np.random.uniform(time_points_age[idx], total_years+time_points_age[0]-0.17)
                label_dict[sample_index] = [subject_index, (start_point-rotations[0])/4.5, (stop_obs-rotations[0])/4.5,
                                            0 if not event_var or idx+1!=len(rotations) else 1, gender, loc_var,
                                            time_points_age[idx],
                                            stop_age,
                                            rotation, rotation,
                                            start_point-rotations[0], digit]

                sample_index += 1

            print(subject_index)
            subject_index += 1
            if sample_index==21:
                pass


    print(dead_number)
    save_data_pd(os.path.join(destination, data_file_name), os.path.join(destination, mask_file_name),
                 os.path.join(destination, data_masked_file_name), os.path.join(destination, labels_file_name),
              rotated_MNIST, label_dict, missing_frac, observed_frac)
    # save_data(data_file, mask_file, data_masked_file, label_file,
    #           rotated_MNIST, label_dict, missing_frac, observed_frac)
    rotated_MNIST = np.empty((0, 1296))
    label_dict = {}


    def create_readme(file_folder, missingness, event_ratio, noise_info):
        with open(f'{file_folder}/README.md', 'w') as file:
            file.write("# Dataset Information\n\n")
            file.write("## Missingness\n")
            file.write(f"{missingness}\n\n")
            file.write("## Event Ratio\n")
            file.write(f"{event_ratio}\n\n")
            file.write("## Noise Information\n")
            file.write(f"{noise_info}\n")
            file.write("## Observation time Information\n")
            file.write(f"Observation time is 6 years in total. The observations are in varying time.\n")


    # Example usage
    missingness = f"{missing}% of the data is missing."
    event_ratio = f"The dataset has a {int(event_prob*100)}:{int(np.round((1-event_prob)*100))} event to non-event ratio."
    noise_info = f"Additive Noise is set to {additive_noise}, and there is shifting."

    create_readme(destination, missingness, event_ratio, noise_info)


    print('Saved! Number of samples: {}'.format(sample_index))
