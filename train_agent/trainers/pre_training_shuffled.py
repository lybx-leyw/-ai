from model.tiny_seek import TinySeek
from tools import Vocab

import torch
import random
import numpy as np
import json
from train_agent.trainers.pre_training import train_epoch
import torch.optim as optim

def pre_train_shuffled(
        config,vocab_data_path,vocab_trg_path,
        json_data_path,max_len,batch_size,
        max_epochs=1,num_workers=4,accumulation_steps=8,
        warmup_index=471,keep_temp_index=1300,
        sava_frq=10,last_index=0,n_of_samples=3000,
        n_of_samplings=5,print_frq=20,
        conlude_epoch=4,init_lr=5e-4,
        seed=42,prefetch_factor=2
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
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    if config['device'] == 'cuda':
        scaler = torch.amp.GradScaler("cuda")

    # 随机抽取n条数据进行经验回放训练
    tran_idx = list(range(1,1413000))
    random.shuffle(tran_idx)
    for input_number in range(n_of_samplings):
        all_ids = []
        # 采样n_of_samples条数据进行训练
        if input_number*n_of_samples >= 1413000:
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
                data = json.loads(line)
                conversations = data['text']
                ids = vocab.encode(conversations,max_len=max_len)
                all_ids.append(ids)
            all_data = torch.cat(all_ids,dim=0)
            train_epoch(
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
                prefetch_factor=prefetch_factor
            )
            all_ids = []
    torch.save(model.state_dict(),f"out\\TinySeek_Pre_final.pkl")