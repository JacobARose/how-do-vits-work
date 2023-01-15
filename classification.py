#!/usr/bin/env python
# coding: utf-8

# # Classification on CIFAR and ImageNet

# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys

# check whether run in Colab
root = "."
if "google.colab" in sys.modules:
    print("Running in Colab.")
    get_ipython().system('pip3 install matplotlib')
    get_ipython().system('pip3 install einops==0.3.0')
    get_ipython().system('pip3 install timm==0.4.9')
    get_ipython().system('git clone -b transformer git@github.com:JacobARose/how-do-vits-work.git')
    root = "./how-do-vits-work"
    sys.path.append(root)


# In[4]:


import os
import time
import yaml
import copy
from pathlib import Path
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import ops.trains as trains
import ops.tests as tests
import ops.datasets as datasets
import ops.schedulers as schedulers


# In[5]:


# config_path = "%s/configs/cifar10_vit.yaml" % root
config_path = "%s/configs/cifar100_vit.yaml" % root
# config_path = "%s/configs/imagenet_vit.yaml" % root

with open(config_path) as f:
    args = yaml.load(f)
    print(args)


# In[6]:


dataset_args = copy.deepcopy(args).get("dataset")
train_args = copy.deepcopy(args).get("train")
val_args = copy.deepcopy(args).get("val")
model_args = copy.deepcopy(args).get("model")
optim_args = copy.deepcopy(args).get("optim")
env_args = copy.deepcopy(args).get("env")


# In[7]:


dataset_train, dataset_test = datasets.get_dataset(**dataset_args, download=True)
dataset_name = dataset_args["name"]
num_classes = len(dataset_train.classes)

dataset_train = DataLoader(dataset_train, 
                           shuffle=True, 
                           num_workers=train_args.get("num_workers", 4), 
                           batch_size=train_args.get("batch_size", 128))
dataset_test = DataLoader(dataset_test, 
                          num_workers=val_args.get("num_workers", 4), 
                          batch_size=val_args.get("batch_size", 128))

print("Train: %s, Test: %s, Classes: %s" % (
    len(dataset_train.dataset), 
    len(dataset_test.dataset), 
    num_classes
))


# ## Model

# Use provided models:

# In[ ]:


# ResNet
# name = "resnet_dnn_50"
# name = "resnet_dnn_101"

# ViT
name = "vit_ti"
# name = "vit_s"

vit_kwargs = {  # for CIFAR
    "image_size": 32, 
    "patch_size": 2,
}

model = models.get_model(name, num_classes=num_classes, 
                         stem=model_args.get("stem", False), **vit_kwargs)
# models.load(model, dataset_name, uid=current_time)

# Or use `timm`:

import timm

model = timm.models.vision_transformer.VisionTransformer(
    img_size=32, patch_size=2, num_classes=num_classes,  # for CIFAR
    embed_dim=192, depth=12, num_heads=3, qkv_bias=False,  # ViT-Ti
)
model.name = "vit_ti"
models.stats(model)


# Parallelize the given `moodel` by splitting the input:

# In[12]:


name = model.name
model = nn.DataParallel(model)
model.name = name


# ## Train

# Define a TensorBoard writer:

# In[13]:


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("runs", dataset_name, model.name, current_time)
writer = SummaryWriter(log_dir)

with open("%s/config.yaml" % log_dir, "w") as f:
    yaml.dump(args, f)
with open("%s/model.log" % log_dir, "w") as f:
    f.write(repr(model))

print("Create TensorBoard log dir: ", log_dir)


# Train the model:

# In[ ]:


gpu = torch.cuda.is_available()
optimizer, train_scheduler = trains.get_optimizer(model, **optim_args)
warmup_scheduler = schedulers.WarmupScheduler(optimizer, len(dataset_train) * train_args.get("warmup_epochs", 0))

trains.train(model, optimizer,
             dataset_train, dataset_test,
             train_scheduler, warmup_scheduler,
             train_args, val_args, gpu,
             writer, 
             snapshot=-1, 
             dataset_name=dataset_name, 
             uid=current_time)  # Set `snapshot=N` to save snapshots every N epochs.


# Save the model:

# In[17]:


models.save(model, dataset_name, current_time, optimizer=optimizer)


# ## Test

# In[18]:


gpu = torch.cuda.is_available()

model = model.cuda() if gpu else model.cpu()
metrics_list = []
for n_ff in [1]:
    print("N: %s, " % n_ff, end="")
    *metrics, cal_diag = tests.test(model, n_ff, dataset_test, verbose=False, gpu=gpu)
    metrics_list.append([n_ff, *metrics])

leaderboard_path = os.path.join("leaderboard", "logs", dataset_name, model.name)
Path(leaderboard_path).mkdir(parents=True, exist_ok=True)
metrics_dir = os.path.join(leaderboard_path, "%s_%s_%s.csv" % (dataset_name, model.name, current_time))
tests.save_metrics(metrics_dir, metrics_list)


# In[16]:


85/15


# In[ ]:




