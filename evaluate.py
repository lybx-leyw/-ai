from evaluate_agent.evaluate_function import evaluate_model

if __name__ == '__main__':
    from tools import ConfigManager
    config = ConfigManager('model_config.json').config
    config['device'] = 'cpu'
    evaluate_model(
        config=config,
        vocab_data_path="dataset\minimind_dataset\pretrain_hq.jsonl",
        vocab_trg_path="vocab.json",
        max_len=512,
        repetition_penalty = 4,
        load_index="0_1_25",
        LoRA=False,
        SFT=False,
        Full=True
    )