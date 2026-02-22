from tools import ConfigManager
config = ConfigManager('model_config.json').config

from train_agent.trainers import prefix_train
import sys

def prefix_train():        
    from torch.multiprocessing import freeze_support
    freeze_support()
    sys.exit(prefix_train(
            config=config,
            vocab_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
            vocab_trg_path="vocab.json",
            json_data_path="dataset\minimind_dataset\sft_mini_512.jsonl",
            max_len=512,
            batch_size=16,
            max_epochs=1,
            num_workers=4,
            accumulation_steps=4,
            warmup_index = 122,
            keep_temp_index = 900,
            sava_frq=50,
            last_index=-1,
            n_of_samples=1000,
            n_of_samplings = 1214,
            print_frq=3,
            conlude_epoch=10,
            seed=42,
            prefetch_factor=6,
            init_lr=4e-5,
            pretraining_model_path="out\\TinySeek_Pre_final.pkl"
        ))