from tools import ConfigManager
config = ConfigManager('model_config.json').config

from train_agent.trainers import lora_train
import sys

def lora_train():    
    from torch.multiprocessing import freeze_support
    freeze_support()
    sys.exit(lora_train(
            config=config,
            vocab_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
            vocab_trg_path="vocab.json",
            json_data_path="dataset\TinySeek_dataset\lora_identity.jsonl",
            max_len=512,
            batch_size=32,
            max_epochs=50,
            num_workers=4,
            accumulation_steps=2,
            warmup_index = -1,
            keep_temp_index = -1,
            sava_frq=10,
            last_index=-1,
            n_of_samples=180,
            n_of_samplings = 1,
            print_frq=1,
            conlude_epoch=10,
            seed=42,
            prefetch_factor=6,
            init_lr=5e-4,
            rank=64,
            pretraining_model_path="out\\TinySeek_Pre_final.pkl"
        ))