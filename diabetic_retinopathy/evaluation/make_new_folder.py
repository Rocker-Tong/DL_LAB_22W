import os
import shutil


def make_folder(path):
    grad_cam_path = path + "/images/grad_cam/"
    if os.path.exists(grad_cam_path):
        shutil.rmtree(grad_cam_path)
    os.makedirs(grad_cam_path)

    guided_backpropagation_path = path + "/images/guided_backpropagation/"
    if os.path.exists(guided_backpropagation_path):
        shutil.rmtree(guided_backpropagation_path)
    os.makedirs(guided_backpropagation_path)

    guided_grad_cam_path = path + "/images/guided_grad_cam/"
    if os.path.exists(guided_grad_cam_path):
        shutil.rmtree(guided_grad_cam_path)
    os.makedirs(guided_grad_cam_path)