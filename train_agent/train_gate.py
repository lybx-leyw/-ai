from train_agent.trainers.gated_training import gated_train
from tools import ConfigManager
config = ConfigManager('model_config.json').config
import sys

def train_gate():    
    from torch.multiprocessing import freeze_support
    freeze_support()
    sys.exit(gated_train(
        config=config,
        vocab_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
        vocab_trg_path="vocab.json",
        json_data_path="dataset\TinySeek_dataset\mix_gated_shuffled.jsonl",
        max_len=512,
        batch_size=32,
        max_epochs=30,
        num_workers=4,
        accumulation_steps=2,
        warmup_index = -1,
        keep_temp_index = -1,
        sava_frq=25,
        last_index=-1,
        n_of_samples=11628,
        n_of_samplings = 1,
        print_frq=1,
        conlude_epoch=10,
        seed=42,
        prefetch_factor=6,
        init_lr=5e-4,
        rank=64
    ))