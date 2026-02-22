from model.tiny_seek import TinySeek
from model.lora_model import apply_lora
from model.tiny_seek import gated_TinySeek
from train_agent.trainers.lora_tuning import LoraDataset
from tools import Vocab

import time
import torch
import torch.nn as nn
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

def gated_train(
        config,vocab_data_path,vocab_trg_path,
        json_data_path,max_len,batch_size,
        max_epochs=1,num_workers=4,accumulation_steps=8,
        warmup_index=471,keep_temp_index=1300,
        sava_frq=10,last_index=0,n_of_samples=3000,
        n_of_samplings=5,print_frq=20,
        conlude_epoch=4,init_lr=5e-4,
        seed=42,prefetch_factor=2,rank=64
        ): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    vocab = Vocab(vocab_data_path,vocab_trg_path,config['vocab_size']) 
    model_1 = TinySeek(config).to(config['device'])
    model_2 = TinySeek(config).to(config['device'])
    model_3 = TinySeek(config).to(config['device'])
    model_4 = TinySeek(config).to(config['device'])
    model_5 = TinySeek(config).to(config['device'])
    model_6 = TinySeek(config).to(config['device'])
    gate_model = TinySeek(config).to(config['device'])
    apply_lora(model=model_1,rank=rank,scale=0.02,prefix=False)
    apply_lora(model=model_2,rank=rank,scale=0.02)
    apply_lora(model=model_3,rank=rank,scale=0.02)
    apply_lora(model=model_4,rank=rank,scale=0.02)
    apply_lora(model=model_5,rank=rank,scale=0.02)
    path_1 = "final\TinySeek_Lora_prefix.pkl"
    path_2 = "final\TinySeek_Lora_medical.pkl"
    path_3 = "final\TinySeek_Lora_r.pkl"
    path_4 = "final\TinySeek_Lora_role.pkl"
    path_5 = "final\TinySeek_Lora_shopl.pkl"
    path_6 = "final\TinySeek_SFT_1.pkl"
    path_g = "final\TinySeek_Pre_final.pkl"
    def load_model(model,path):
        try:
            data = torch.load(path, map_location=config['device'])
            if isinstance(data, dict):
                model.load_state_dict(data)
            else:
                try:
                    model.load_state_dict(data.state_dict())
                except Exception:
                    try:
                        state_dict = torch.load(path, map_location=config['device'])
                        with open("model_layers.txt","a",encoding='utf-8') as file:
                            file.write(f'{getattr(state_dict, "keys", lambda: "<no-keys>")()}')
                        model.load_state_dict(state_dict)
                    except Exception as e:
                        print(f"无法加载模型文件 {path}: {e}")
                        raise
            model.to(config['device'])
            print(f"成功加载模型参数: {path}")
        except FileNotFoundError:
            print(f"未找到模型参数:{path}")
        for param in model.parameters():
            param.requires_grad = False
            
    load_model(model_1,path_1)
    load_model(model_2,path_2)
    load_model(model_3,path_3)
    load_model(model_4,path_4)
    load_model(model_5,path_5)
    load_model(model_6,path_6)
    load_model(gate_model,path_g)
    apply_lora(gate_model, rank=rank, scale=0.02)

    model = gated_TinySeek(seq_len=max_len,config=config,rank=rank).to(config['device'])
    model.load_part_model(
        model_1=model_1,model_2=model_2,model_3=model_3,
        model_4=model_4,model_5=model_5,model_6=model_6,
        gate_model=gate_model
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {trainable_params}")
    optimizer = optim.Adam(model.parameters(),init_lr)
    if config['device'] == 'cuda':
        scaler = torch.amp.GradScaler("cuda")
    scheduler = True

    tran_idx = list(range(1,n_of_samples*n_of_samplings))
    random.shuffle(tran_idx)
    for input_number in range(n_of_samplings):
        all_ids = []
        # 采样n_of_samples条数据进行训练
        if input_number*n_of_samples >= n_of_samples*n_of_samplings:
            print("已完成所有数据的采样训练")
            break
        # 跳过已经训练过的数据
        if input_number*n_of_samples <= last_index:
            continue
        random_indices = set(tran_idx[input_number*n_of_samples:(input_number+1)*n_of_samples])
        with open(json_data_path, "r", encoding='utf-8') as data_file:
            for line_index, line in enumerate(data_file, 1):
                if line_index not in random_indices:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"出错的行: {line[:200]}")  # 打印前200个字符
                    print(f"行号: {line_index}")
                    print(f"错误位置: {e}")
                    raise
                text = ""
                for turn in data['conversations']:
                    if turn['role'] == 'user':
                        text += f"User：{turn['content']}\n"
                    else:
                        text += f"Assistant：{turn['content']}<|im_end|>\n"
                ids = vocab.encode(text,max_len=max_len)
                all_ids.append(ids)
            if len(all_ids) == 0:
                continue
            all_data = torch.cat(all_ids,dim=0)
            gate_train_epoch(
                config=config,
                data=all_data,
                batch_size=batch_size,
                max_epochs=max_epochs,
                num_workers=num_workers,
                model=model,
                optimizer=optimizer,
                scheduler=True,
                number=input_number*n_of_samples,
                accumulation_steps=accumulation_steps,
                scaler=scaler if config['device'] == 'cuda' else None,
                warmup_index=warmup_index,
                keep_index=keep_temp_index,
                save_frq=sava_frq,
                print_frq=print_frq,
                conlude_epoch=conlude_epoch,
                n_of_samples=n_of_samples,
                n_of_sampling=n_of_samplings,
                prefetch_factor=prefetch_factor,
                scale=0.1
            )
            all_ids = []
    torch.save(model.state_dict(),f"out\\TinySeek_gate_final.pkl")

def gate_train_epoch(config,data,batch_size,max_epochs,
                num_workers,model,optimizer,scheduler,
                number,accumulation_steps=8,scaler=None,
                warmup_index=471,keep_index=1300,save_frq=10,print_frq=20,
                conlude_epoch=4,n_of_samples=300,n_of_sampling=4710,
                prefetch_factor=2,scale=1,init_lr=0.1):
    if number//n_of_samples <= keep_index:
        scheduler = False
    optimizer.zero_grad()
    device = config['device']
    dataset = LoraDataset(data,pad_idx=config['pad_idx'],mask_start_id=9,mask_end_id=10)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        drop_last=True,  
        pin_memory=True,       
        prefetch_factor=prefetch_factor,    
        num_workers=num_workers,
        persistent_workers=True
    )
    # 开始训练
    for epoch in range(max_epochs):
        timer_epoch = time.perf_counter()
        total_acc = 0
        for batch_idx,(batch_src,batch_trg_1,batch_trg_2) in enumerate(train_dataloader):
            timer = time.perf_counter()

            batch_src = batch_src.to(device)
            batch_trg_1 = batch_trg_1.to(device)
            batch_trg_2 = batch_trg_2.to(device)

            model.train()
            if scaler is not None:
                with torch.amp.autocast('cuda',dtype=torch.float16):  
                    outputs_1,outputs_2,Lexp = model(batch_src)
                    cross_loss_1 = F.cross_entropy(
                        outputs_1.reshape(-1,config['vocab_size']),
                        batch_trg_1.reshape(-1),
                        ignore_index = config['pad_idx']
                    )
                    cross_loss_2 = F.cross_entropy(
                        outputs_2.reshape(-1,config['vocab_size']),
                        batch_trg_2.reshape(-1),
                        ignore_index = config['pad_idx']
                    )
                    cross_loss = cross_loss_1 + cross_loss_2
                    loss = cross_loss + config['alpha']*Lexp  
                    train_loss,acc = model.cacu_gate_loss(batch_trg_1,batch_trg_2,config,batch_src)                
                    train_loss = train_loss/accumulation_steps  
                    scaler.scale(train_loss).backward()    
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is True:
                        # 指数衰减学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = max(param_group['lr']*0.9995,1e-7)
                    else:
                        # warmup阶段使用线性增长的学习率,keep阶段保持学习率不变
                        current_index = number//n_of_samples
                        warmup_factor = min(1.0, current_index/warmup_index)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = max(init_lr*warmup_factor,1e-7)
            else:
                outputs_1,outputs_2,Lexp = model(batch_src)
                outputs_1 = outputs_1*scale
                outputs_2 = outputs_2*scale
                cross_loss_1 = F.cross_entropy(
                    outputs_1.reshape(-1,config['vocab_size']),
                    batch_trg_1.reshape(-1),
                    ignore_index = config['pad_idx']
                )
                cross_loss_2 = F.cross_entropy(
                    outputs_2.reshape(-1,config['vocab_size']),
                    batch_trg_2.reshape(-1),
                    ignore_index = config['pad_idx']
                )
                cross_loss = cross_loss_1 + cross_loss_2
                loss = cross_loss + config['alpha']*Lexp                    
                train_loss,acc = model.cacu_gate_loss(batch_trg_1,batch_trg_2,config,batch_src)                
                train_loss = train_loss/accumulation_steps  
                train_loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is True:
                        # 指数衰减学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = max(param_group['lr']*0.9995,1e-7)
                    else:
                        # warmup阶段使用线性增长的学习率,keep阶段保持学习率不变
                        current_index = number//n_of_samples
                        warmup_factor = min(1.0, current_index/warmup_index)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = max(init_lr*warmup_factor,1e-7)

            perplexity_1 = torch.exp(cross_loss_1)
            perplexity_2 = torch.exp(cross_loss_2)
            current_lr = optimizer.param_groups[0]['lr']

            timer_end = time.perf_counter()
            time_batch = timer_end-timer
            last_batch_time = time_batch*(len(train_dataloader)-batch_idx-1)
            last_index_time = time_batch*len(train_dataloader)*(n_of_sampling-number//n_of_samples-1)
            last_time = last_batch_time+last_index_time

            total_acc += acc
            if (batch_idx+1) % print_frq == 0:   
                log_line = (f'Epoch {epoch+1:03d}/{max_epochs} | '
                            f'Batch {batch_idx+1:04d}/{len(train_dataloader)} | '
                            f'Index {number//n_of_samples:04d}/{n_of_sampling} | '
                            f'Train_loss: {train_loss*accumulation_steps:8.2f} | '
                            f'Ppx_1: {perplexity_1.item():8.2f} | '
                            f'Ppx_2: {perplexity_2.item():8.2f} | '
                            f'LR: {current_lr:8.4e} | '
                            f'Loss: {loss:8.4f} | '
                            f'Acc: {total_acc/(batch_idx+1):4.2f} | '
                            f'Device: {device} | '
                            f'Batch_t: {time_batch:5.2f}s | '
                            f'Remaining: {last_time/3600:5.2f}h')
                print(log_line)
                with open("log.txt","a",encoding='utf-8') as log:
                    log.write(f'{log_line}\n')
            if (batch_idx+1) % save_frq == 0:   
                torch.save(model.state_dict(),f"out\\TinySeek_gate{number}_{epoch+1}_{batch_idx+1}.pkl")
        
        timer_epoch_end = time.perf_counter()
        epoch_time = timer_epoch_end-timer_epoch
        last_index_time = epoch_time*(n_of_sampling-number//n_of_samples-1)
        if number//n_of_samples % conlude_epoch == 0:
            print(f"| Estimated Remaining:{last_index_time/3600:.2f} hours | ")
            with open("log.txt","a",encoding='utf-8') as log:
                log.write(f"| Estimated Remaining:{last_index_time/3600:.2f} hours | \n")