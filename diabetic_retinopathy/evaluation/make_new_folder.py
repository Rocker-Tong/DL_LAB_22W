import os
import shutil


def make_folder(path):
    grad_cam_path = path + "/images/grad_cam/"
    if os.path.exists(grad_cam_path):
        shutil.rmtree(grad_cam_path)
    os.makedirs(grad_cam_path)
    grad_cam_backpropagation_path = path + "/images/guided_backpropagation/"
    if os.path.exists(grad_cam_backpropagation_path):
        shutil.rmtree(grad_cam_backpropagation_path)
    os.makedirs(grad_cam_backpropagation_path)