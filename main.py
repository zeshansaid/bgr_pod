import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup

from rp_schema import INPUT_SCHEMA


import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageFile
from backgroundremover import utilities
from backgroundremover.bg import remove
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os 
import base64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



def upload_or_base64_encode(file_name, img_path):
    """
    Uploads image to S3 bucket if it is available, otherwise returns base64 encoded image.
    """
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        return upload_file_to_bucket(file_name, img_path)

    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")



# -------------------------------------------- ------------------ -----------------------------------------
# -------------------------------------------- BackGround Removal -----------------------------------------
# -------------------------------------------- ------------------ -----------------------------------------

model_choices = ["u2net", "u2net_human_seg", "u2netp"]
bgr_model = "u2net"
alpha_matting = False
alpha_matting_foreground_threshold = 24 # 240# The trimap foreground threshold.
alpha_matting_background_threshold = 10 # The trimap background threshold.
alpha_matting_erode_size = 10# Size of element used for the erosion.
alpha_matting_base_size = 1000 # The image base size.
workernodes = 8#1 # Number of parallel workers
gpubatchsize = 260#2 # GPU batchsize
framerate = -1 # override the frame rate
framelimit = -1 # Limit the number of frames to process for quick testing.
mattekey = False # Output the Matte key file , type=lambda x: bool(strtobool(x)),
transparentvideo = False # Output transparent video format mov
transparentvideoovervideo = False # Overlay transparent video over another video
transparentvideooverimage = False # Overlay transparent video over another video
transparentgif = False # Make transparent gif from video
transparentgifwithbackground = False # Make transparent background overlay a background image

def process(data):
    base64Image= data
    new_image = remove(
                    base64Image,
                    model_name=bgr_model,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_structure_size=alpha_matting_erode_size,
                    alpha_matting_base_size=alpha_matting_base_size,
                )
    return new_image #end of function process


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']
    job_input = job_input['image']
    job_input = process(job_input)
    return job_input #end of run function


if __name__ == "__main__":
    runpod.serverless.start({"handler": run})