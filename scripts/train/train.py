import os
import json
import time

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt

from model import protonet
from data_process import transform_data

def run(arg):
    if os.path.isdir(arg["--log_dir"])==False:
        os.mkdir(arg["--log_dir"])

    with open(os.path.join(arg["--log_dir"],"arg.json"),'w') as f:
        json.dump(arg,f)
        f.write('\n')

    if os.path.isdir(arg["--model_savedir"])==False:
        os.mkdir(arg["--model_savedir"])

    model_savedir=arg["--model_savedir"]
    log_file=arg["--log_dir"]+"log_file.txt"

    if arg["--dataset"]!="mini-Imagenet":
        raise ValueError("Unknown dataset {:s}".format(arg["--dataset"]))

    dataset=transform_data.extract_data(arg["--data_split"],arg["--data_dir"])
    protonet_model=protonet(arg["--input_dimensionality"],arg["--hidden_dimensionality"],arg["--output_dimensionality"])
    
    if(arg["--cuda"]):
        protonet_model.cuda()

    optimizer=getattr(optim,arg["--train_optim"])(protonet_model.parameters(),lr=arg["--train_lr"]
                        ,weight_decay=arg["--train_weight_decay"])

    scheduler=lr_scheduler.StepLR(optimizer,arg["--train_decay"],gamma=0.5)
    log=[]
    epoch=arg["--train_epoch"]

    train_loader=DataLoader(dataset["train"],shuffle=True,drop_last=True)
    val_loader=DataLoader(dataset["val"],shuffle=True,drop_last=True)

    if os.path.isfile(log_file):
        os.remove(log_file)

    start_time=time.time()
    best_loss=10000
    patience=arg["--train_patience"]
    for i in range(0,epoch):
        scheduler.step()
        avg_loss=0
        avg_acc=0
        train_iters_time=0
        epoch_start_time=time.time()
        #training iterations
        protonet_model.train()
        log_text=""
        for samples in train_loader:
            optimizer.zero_grad()
            loss,acc=protonet_model.forward(samples)
            avg_loss+=loss
            avg_acc+=acc
            train_iters_time+=1
            loss.backward()
            optimizer.step()

        avg_loss/=train_iters_time
        avg_acc/=train_iters_time
        if val_loader is not None:
            protonet_model.eval()
            val_avg_loss=0
            val_iters_time=0
            val_avg_acc=0
            for samples in val_loader:
                loss,acc=protonet_model.forward(samples)
                val_avg_loss+=loss
                val_avg_acc+=acc
                val_iters_time+=1

            val_avg_loss/=val_iters_time
            val_avg_acc/=val_iters_time
            epcoh_end_time=time.time()
            if val_avg_loss<best_loss:
                patience=arg["--train_patience"]
                best_loss=val_avg_loss
                protonet_model.cpu()
                torch.save(protonet_model,os.path.join(model_savedir,"best_model_.pt"))
                if arg["--cuda"]:
                    protonet_model.cuda()

            else:
                patience-=1

            log_text="Epoch {:d}:train accuracy:{:f},loss:{:f} validation accuracy:{:f},loss{:f}".format(i,avg_acc,avg_loss,
                            val_avg_acc,val_avg_loss)            
        else:
            log_text="Epoch {:d}:train accuarcy:{:f},loss{:f}".format(i,avg_acc,avg_loss)

        print(log_text)

        log.append(log_text)

        if patience==0:
            break

    torch.save(protonet_model,os.path.join(model_savedir,"final.pt"))        

    end_time=time.time()
    training_time="Total training time:{:f}".format(end_time-start_time)
    log.append(training_time)
    with open(log_file,'w') as f:
        f.write('\n'.join(log))

    print(training_time)
