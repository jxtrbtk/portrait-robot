# ###############################################79############################
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module 



"""
# #############79##############################################################
#                                      #
__author__ = "jxtrbtk"                 #
__contact__ = "bYhO-bOwA-dIcA"         #
__date__ = "JaXy-QeVi-Ka"              # Tue Jan 29 12:56:06 2019
__email__ = "j.t[4t]free.fr"           #
__version__ = "1.0"                    #
#                                      #
# ##################################79#########################################
import os
import io
import time
import requests
import json
import torch
import cv2
import pickle 
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import save_image

import model

import SyllabInt


PARAM_DATA_FOLDER = "data"
PARAM_HOST_BASE_URL = "http://localhost/portrait/"

GENERATOR = None


def Main():
    Load_Generator()
    url = PARAM_HOST_BASE_URL + "cTaskGetList.php"
    r = requests.get(url)
    tasks = r.json()
    for task in tasks:
        task_id = task.split(".")[0]
        Execute_Task(task_id)


def Load_Generator():
    global GENERATOR
    GENERATOR = model.Generator(z_size=model.z_size, conv_dim=model.g_conv_dim)
    model_path = os.path.join("data", "model", "ModelG_49.pth")
    generator_state_dict = torch.load(model_path, map_location='cpu')
    GENERATOR.load_state_dict(generator_state_dict)
    GENERATOR.eval()  # for generating images


def Check_Pending_Work():
    url = PARAM_HOST_BASE_URL + "cTaskGetList.php"
    r = requests.get(url)
    tasks = r.json()
    return (len(tasks) > 0)


def Execute_Task(task_id):
    print("task : {}".format(task_id))
    url = PARAM_HOST_BASE_URL + "cTaskGet.php?id="+task_id
    r = requests.get(url)
    command = r.json()
    params = ""

    if command["action"] == "new":
        s = SyllabInt.SyllabInt()
        code = s.code
        time.sleep(1)
        print("  code: {}".format(code))
        Initialize_Project(code)
        print(" - initialized")
        Generate_Images(code)
        print(" - generated")
        params += "&code=" + code
        params += "&step=0"

    if command["action"] == "check":
        code = command["code"]
        print("  code: {}".format(code))
        step = Generate_Images(code)
        print(" - generated")
        params += "&code=" + code
        params += "&step=" + str(step)

    if command["action"] == "select":
        code = command["code"]
        side = command["side"]
        print("  code: {}".format(code))
        print("  side: {}".format(side))
        step = Move_Project(code, side)        
        print(" - moved")
        Generate_Images(code)
        print(" - generated")
        params += "&code=" + code
        params += "&step=" + str(step)

    url = PARAM_HOST_BASE_URL + "cTaskFeedback.php?id="+task_id + params
    requests.get(url)

def Move_Project(code, side):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)

    filepath = os.path.join(path, "step.txt")
    step = Read_File(filepath)
    step = int(step) +1
    Write_File(filepath, str(step))

    filepath = os.path.join(path, "vectors.p")
    vectors = pickle.load(open(filepath, "rb"))
    vector0 = vectors[int(side)-1]

    sample_size=1
    coeff = (0.99**step)
    print(coeff)

    delta = np.random.uniform(-1, 1, size=(sample_size, model.z_size))
    delta = torch.from_numpy(delta).float()
    delta[delta<0] = vector0[delta<0]
    vector2 = vector0*(1-coeff)+delta*coeff
#    vector1 = vector0*(0.5)+vector2*(0.5)

    delta = np.random.uniform(-1, 1, size=(sample_size, model.z_size))
    delta = torch.from_numpy(delta).float()
    delta[delta<0] = vector0[delta<0]
    vector1 = vector0*(1-coeff)+delta*coeff
    
#    delta = delta * (-1.0)
#    vector1 = vector0*(1-coeff)+delta*coeff

    vectors = (vector1, vector0, vector2)
    pickle.dump(vectors, open(filepath, "wb"))

    return step



#    pickle.dump(vectors, open(filepath, "wb"))


def Generate_Images(code):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)
    filepath = os.path.join(path, "step.txt")
    step = Read_File(filepath)
    filepath = os.path.join(path, "vectors.p")
    vectors = pickle.load(open(filepath, "rb"))
    with torch.no_grad():
        for idx, vector in enumerate(vectors):
            img = GENERATOR(vector)
            img_name = "IMG_{}_{}_{}.png".format(idx+1, code, step)
            img_path = os.path.join(path, img_name)
            save_image(img, img_path, normalize=True)
            Post_Image(idx+1, code, step)
    return step

def Post_Image(idx, code, step):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)
    img_name = "IMG_{}_{}_{}.png".format(idx, code, step)
    img_path = os.path.join(path, img_name)
    payload = {'name': img_name, 'side': idx, 'code': code, 'step': step}
    url = PARAM_HOST_BASE_URL + "cImgPost.php"
    with io.open(img_path, 'rb') as f: 
        r = requests.post(url, data=payload, files={'data': f})
        print(r.text)
#        if(r.text=="OK"):
#            os.remove(img_path)


def Initialize_Project(code):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)
    if not os.path.exists(path):
        os.mkdir(path)
    filepath = os.path.join(path, "step.txt")
    Write_File(filepath, "0")
    filepath = os.path.join(path, "vectors.p")
    vectors = (Create_New_Vector(), Create_New_Vector(), Create_New_Vector())
    pickle.dump(vectors, open(filepath, "wb"))


def Create_New_Vector():
    sample_size=1
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, model.z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    return fixed_z

def Write_File(filepath, line):
    with io.open(filepath, "w") as f:
        f.write(str(line))


def Write_To_File(filepath, line):
    with io.open(filepath, "a") as f:
        f.write(str(line)+"\n")


def Read_File(filepath):
    with io.open(filepath, "r") as f:
        data = str(f.read())
    return data


# #################################################################79##########
# local unit tests


if __name__ == "__main__":

    if Check_Pending_Work():
        for i in range(0,100):
            print(i)
            Main()
            time.sleep(3)
 
    
    
# #######################################################79####################
