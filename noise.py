import cv2
import os
import numpy as np

def get_sigma_value(yuv_img):
    
    # Extract the Y channel (luminance)
    y_channel = yuv_img[:, :, 0]

    # Calculate the standard deviation (sigma) of the Y channel
    sigma_value = np.std(y_channel)

    return sigma_value

def get_median_gray_value(yuv_img):  

    # Extract the Y channel (luminance)
    y_channel = yuv_img[:, :, 0]

    # Flatten the 2D array to a 1D array
    flattened_array = y_channel.flatten()

    # Calculate the median value
    median_value = np.median(flattened_array)

    return median_value


def denoise_y_channel(y_channel):
    # Apply fastNlMeansDenoising to denoise the Y channel
    denoised_y = cv2.fastNlMeansDenoising(y_channel, None, h=5, templateWindowSize=4, searchWindowSize=4)
    return denoised_y



def convert_to_yuv_and_denoise_before_writing_as_rgb(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Read the image
        image_path = os.path.join(input_folder, file)
        img = cv2.imread(image_path)

                
        # Convert the image to YUV color scheme
        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


        y_channel = yuv_img[:, :, 0]

        # Denoise the Y channel
        denoised_y = denoise_y_channel(y_channel)

        yuv_img[:, :, 0] = denoised_y;

        rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)


        # get statistical values from yuv_img
        #median_grey_value = get_median_gray_value(yuv_img);
        #sigma_value = get_sigma_value(yuv_img);       

        #th = (sigma_value*2) + median_grey_value/60;

        #print(th);

        # Save the YUV image to the output folder
        output_path = os.path.join(output_folder, f'rgb_{file}')
        cv2.imwrite(output_path, rgb_img)

        #print(f'Converted {file} to YUV and saved to {output_path}, median gray value = {median_grey_value}, sigma value = {sigma_value}')

# Replace 'input_folder' and 'output_folder' with your actual folder paths
input_folder = './img'
output_folder = './out'

convert_to_yuv_and_denoise_before_writing_as_rgb(input_folder, output_folder)
