from model.tiny_seek import TinySeek
from tools import Vocab

import time
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

class PreTrainingDataset(Dataset):
    def __init__(self,text_tensor,pad_idx):
        self.input_ids = text_tensor
        self.pad_idx = pad_idx
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):
        input_seq = self.input_ids[idx]
        labels_1 = torch.roll(input_seq,shifts=-1,dims=-1)
        labels_1[-1] = self.pad_idx     

        labels_2 = torch.roll(labels_1,shifts=-1,dims=-1)
        labels_2[-1] = self.pad_idx     
        
        return input_seq,labels_1,labels_2


def pre_train(
        config,vocab_data_path,vocab_trg_path,
        json_data_path,max_len,batch_size,
        max_epochs=1,num_workers=4,accumulation_steps=8,
        warmup_index=471,keep_temp_index=1300,
        sava_frq=10,last_index=0,print_frq=20,
        conlude_epoch=4,seed=42
        ): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vocab = Vocab(vocab_data_path,vocab_trg_path,config['vocab_size']) 
    model = TinySeek(config).to(config['device'])
    # 加载上一次训练的模型参数
    try:
        try:
            model = torch.load(f"out\\TinySeek_Pre{last_index}_1.pkl").to(config['device'])
        except:
            model.load_state_dict(torch.load(f"out\\TinySeek_Pre{last_index}_1.pkl"))
        print(f"成功加载模型参数：TinySeek_Pre{last_index}_1.pkl")
    except FileNotFoundError:
        print(f"未找到模型参数：TinySeek_Pre{last_index}_1.pkl，开始新的训练")
    optimizer = optim.Adam(model.parameters(),5e-4)
    if config['device'] == 'cuda':
        scaler = torch.amp.GradScaler("cuda")
    scheduler = True

    all_ids = []
    with open(json_data_path, "r", encoding='utf-8') as data_file:
        for line_index, line in enumerate(data_file, 1):
            if line_index <= last_index:
                continue
            data = json.loads(line)
            conversations = data['text'].split('<|im_end|>')
            for item in conversations:
                ids = vocab.encode(item,max_len=max_len)
                all_ids.append(ids)
            if line_index % 300 == 0 and all_ids:
                if line_index//300 <= keep_temp_index:
                    scheduler = False
                all_data = torch.cat(all_ids,dim=0)
                train_epoch(
                    config=config,
                    data=all_data,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    num_workers=num_workers,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    number=line_index,
                    accumulation_steps=accumulation_steps,
                    scaler=scaler if config['device'] == 'cuda' else None,
                    warmup_index=warmup_index,
                    keep_index=keep_temp_index,
                    save_frq=sava_frq,
                    print_frq=print_frq,
                    conlude_epoch=conlude_epoch
                )
                all_ids = []
                print(f"第{line_index}行单词训练完毕") 
    torch.save(model.state_dict(),f"out\\TinySeek_Pre_final.pkl")

def train_epoch(config,data,batch_size,max_epochs,
                num_workers,model,optimizer,scheduler,
                number,accumulation_steps=8,scaler=None,
                warmup_index=471,keep_index=1300,save_frq=10,print_frq=20,
                conlude_epoch=4,n_of_samples=300,n_of_sampling=4710,
                prefetch_factor=2):
    if number//n_of_samples <= keep_index:
        scheduler = False
    optimizer.zero_grad()
    device = config['device']
    dataset = PreTrainingDataset(data,pad_idx=config['pad_idx'])
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
        for batch_idx,(batch_src,batch_trg_1,batch_trg_2) in enumerate(train_dataloader):
            timer = time.perf_counter()

            train_loss = 0.0

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
                    train_loss = loss/accumulation_steps  
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
                            param_group['lr'] = 5e-4*warmup_factor
            else:
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
                train_loss = loss/accumulation_steps  
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
                            param_group['lr'] = 5e-4*warmup_factor

            perplexity_1 = torch.exp(cross_loss_1)
            perplexity_2 = torch.exp(cross_loss_2)
            current_lr = optimizer.param_groups[0]['lr']

            timer_end = time.perf_counter()
            time_batch = timer_end-timer
            last_batch_time = time_batch*(len(train_dataloader)-batch_idx-1)
            last_index_time = time_batch*len(train_dataloader)*(n_of_sampling-number//n_of_samples-1)
            last_time = last_batch_time+last_index_time

            if (batch_idx+1) % print_frq == 0:   
                log_line = (f'Epoch {epoch+1:03d}/{max_epochs} | '
                            f'Batch {batch_idx+1:04d}/{len(train_dataloader)} | '
                            f'Index {number//n_of_samples:04d}/{n_of_sampling} | '
                            f'Loss: {loss:8.4f} | '
                            f'Ppx_1: {perplexity_1.item():8.2f} | '
                            f'Ppx_2: {perplexity_2.item():8.2f} | '
                            f'LR: {current_lr:8.4e} | '
                            f'Device: {device} | '
                            f'Batch_t: {time_batch:5.2f}s | '
                            f'Remaining: {last_time/3600:5.2f}h')
                print(log_line)
                with open("log.txt","a",encoding='utf-8') as log:
                    log.write(f'{log_line}\n')
        
        timer_epoch_end = time.perf_counter()
        epoch_time = timer_epoch_end-timer_epoch
        last_index_time = epoch_time*(n_of_sampling-number//n_of_samples-1)
        if number//n_of_samples % conlude_epoch == 0:
            print(f"| Estimated Remaining:{last_index_time/3600:.2f} hours | ")
            with open("log.txt","a",encoding='utf-8') as log:
                log.write(f"| Estimated Remaining:{last_index_time/3600:.2f} hours | \n")
        if number//n_of_samples % save_frq == 0:
            torch.save(model.state_dict(),f"out\\TinySeek_Pre{number}_{epoch+1}.pkl")