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
__version__ = "2.0"                    #
#                                      #
# ##################################79#########################################
import os
import io
import time
import requests
import json
import torch
# import cv2
import pickle 
import numpy as np
# from matplotlib import pyplot as plt
from torchvision.utils import save_image

import model

import SyllabInt


PARAM_DATA_FOLDER = "data"
#PARAM_HOST_BASE_URL = "http://localhost/portrait/"
PARAM_HOST_BASE_URL = "http://yeah.free.fr/portrait/"

GENERATOR = None


def Main():
    Load_Generator()
    url = PARAM_HOST_BASE_URL + "cTaskGetList.php"
    r = requests.get(url, timeout=60)
    tasks = r.json()
    done = 0
    for task in tasks:
        task_id = task.split(".")[0]
        Execute_Task(task_id)
        done = 1
    return done


def Load_Generator():
    global GENERATOR
    GENERATOR = model.Generator(z_size=model.z_size, conv_dim=model.g_conv_dim)
    model_path = os.path.join("data", "model", "ModelG_49.pth")
    generator_state_dict = torch.load(model_path, map_location='cpu')
    GENERATOR.load_state_dict(generator_state_dict)
    GENERATOR.eval()  # for generating images


def Check_Pending_Work():
    url = PARAM_HOST_BASE_URL + "cTaskGetList.php"
    r = requests.get(url, timeout=60)
    tasks = r.json()
    return (len(tasks) > 0)


def Execute_Task(task_id):
    print("task : {}".format(task_id))
    url = PARAM_HOST_BASE_URL + "cTaskGet.php?id="+task_id
    r = requests.get(url, timeout=60)
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
        score = Get_Score(code)
        print(" - generated")
        params += "&code=" + code
        params += "&step=" + str(step)
        params += "&score=" + "{:.4f}".format(score*100)

    if command["action"] == "select":
        code = command["code"]
        side = command["side"]
        print("  code: {}".format(code))
        print("  side: {}".format(side))
        best_score = Is_Best_Score(code, side) 
        print(best_score)
        step = Move_Project(code, side)        
        score = Get_Score(code)
        print(" - moved")
        Generate_Images(code)
        print(" - generated")
        params += "&code=" + code
        params += "&step=" + str(step)
        params += "&score=" + "{:.4f}".format(score*100)
        if best_score : params += "%20+"

    url = PARAM_HOST_BASE_URL + "cTaskFeedback.php?id="+task_id + params
    requests.get(url, timeout=60)

    
def Is_Best_Score(code, side):
    scores = (0.0, Get_Score(code, side=1), Get_Score(code, side=2), Get_Score(code, side=3))
    return  (max(scores) == scores[int(side)])


def Move_Project(code, side):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)

    filepath = os.path.join(path, "step.txt")
    step = Read_File(filepath)
    step = int(step) +1
    Write_File(filepath, str(step))

    filepath = os.path.join(path, "vectors.p")
    vectors = pickle.load(open(filepath, "rb"))
    vector_target   = vectors[0]          
    vector_selected = vectors[int(side)]

    sample_size=1
    coeff = (0.99**step)
    print(coeff)

    mask = np.random.uniform(0, 1.5, size=(sample_size, model.z_size))
    mask = torch.from_numpy(mask).float()

    delta = np.random.uniform(-1, 1, size=(sample_size, model.z_size))
    delta_left  = torch.from_numpy(delta).float()
    delta_right = torch.from_numpy(-delta).float()
    delta_left [mask>coeff] = vector_selected[mask>coeff]
    delta_right[mask>coeff] = vector_selected[mask>coeff]
    vector_left  = vector_selected*(1-coeff) + delta_left  *coeff
    vector_right = vector_selected*(1-coeff) + delta_right *coeff

    vectors = (vector_target, vector_left, vector_selected, vector_right)
    pickle.dump(vectors, open(filepath, "wb"))

    return step

def Generate_Images(code):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)
    filepath = os.path.join(path, "step.txt")
    step = None
    if os.path.exists(filepath):
        step = Read_File(filepath)
        filepath = os.path.join(path, "vectors.p")
        vectors = pickle.load(open(filepath, "rb"))
        with torch.no_grad():
            for idx, vector in enumerate(vectors):
                img = GENERATOR(vector)
                img_name = "IMG_{}_{}_{}.png".format(idx, code, step)
                img_path = os.path.join(path, img_name)
                save_image(img, img_path, normalize=True)
                Post_Image(idx, code, step)
    return step

def Post_Image(idx, code, step):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)
    img_name = "IMG_{}_{}_{}.png".format(idx, code, step)
    img_path = os.path.join(path, img_name)
    payload = {'name': img_name, 'side': idx, 'code': code, 'step': step}
    url = PARAM_HOST_BASE_URL + "cImgPost.php"
    answer = ""
    with io.open(img_path, 'rb') as f: 
        r = requests.post(url, data=payload, files={'data': f}, timeout=60)
        print(r.text)
        answer = r.text
    if(answer =="OK"):
        os.remove(img_path)


def Initialize_Project(code):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)
    if not os.path.exists(path):
        os.mkdir(path)
    filepath = os.path.join(path, "step.txt")
    Write_File(filepath, "0")
    filepath = os.path.join(path, "vectors.p")
    vectors = (Create_New_Vector(), Create_New_Vector(), Create_New_Vector(), Create_New_Vector())
    pickle.dump(vectors, open(filepath, "wb"))

    
def Get_Score(code, side=2):
    path = os.path.join(PARAM_DATA_FOLDER, "work", code)
    filepath = os.path.join(path, "vectors.p")
    score = 0
    if os.path.exists(filepath):
        filepath = os.path.join(path, "vectors.p")
        vectors = pickle.load(open(filepath, "rb"))
        score = float(1-torch.sum((vectors[0]-vectors[side])**2) / 256)
    print("score", side, score)
    return score
    

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
        i = 0 
        while i < 222:
            i += 1
            print(i)
            d = Main()
            if d == 1: i = 0
            time.sleep(7)
 
    
    
# #######################################################79####################
