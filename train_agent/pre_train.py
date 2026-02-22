from tools import ConfigManager
config = ConfigManager('model_config.json').config

from train_agent.trainers.pre_training import pre_train
from train_agent.trainers.pre_training_shuffled import pre_train_shuffled
import sys

def pre_train():       
    from torch.multiprocessing import freeze_support
    freeze_support()
    interactive = input("请选择训练模式：1.顺序训练 2.随机训练（输入数字1或2）:") 
    if interactive == '1': 
        sys.exit(pre_train(
            config=config,
            vocab_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
            vocab_trg_path="vocab.json",
            json_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
            max_len=512,
            batch_size=32,
            max_epochs=1,
            num_workers=4,
            accumulation_steps=2,
            warmup_index=471,
            keep_temp_index = 3000,
            sava_frq=100,
            last_index=-1,
            print_frq=20,
            conlude_epoch=4
        ))
    elif interactive == '2':
        sys.exit(pre_train_shuffled(
            config=config,
            vocab_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
            vocab_trg_path="vocab.json",
            json_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
            max_len=512,
            batch_size=16,
            max_epochs=1,
            num_workers=4,
            accumulation_steps=4,
            warmup_index = 471,
            keep_temp_index = 3000,
            sava_frq=100,
            last_index=-1,
            n_of_samples=300,
            n_of_samplings=4710,
            print_frq=3,
            conlude_epoch=10,
            seed=43,
            prefetch_factor=6,
            init_lr=5e-4
        ))