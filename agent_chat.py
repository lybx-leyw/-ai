from evaluate_agent.chat import start_chat_session

if __name__ == "__main__":
    start_chat_session(
        config_path='model_config.json',
        device='cpu',
        load_index="_final",
        LoRA=False,
        SFT=False,
        Full=True
    )