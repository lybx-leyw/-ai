from model.tiny_seek import TinySeek
from tools import Vocab, print_processing, Tokenizer
import torch
from evaluate_agent.evaluate_function import load_model_with_lora, load_model_with_sft, load_full_model, load_pretrained_model


def interactive_chat(
    config,
    vocab_data_path,
    vocab_trg_path,
    max_len=512,
    repetition_penalty=1.2,
    load_index=0,
    LoRA=False,
    SFT=False,
    Full=False,
    gate_rank=64,
):
    # 加载词表
    vocab = Vocab(vocab_data_path, vocab_trg_path, config['vocab_size'])

    # 初始化模型结构
    if Full:
        from model.tiny_seek import gated_TinySeek
        model = gated_TinySeek(seq_len=max_len, config=config).to(config['device'])
    else:
        model = TinySeek(config).to(config['device'])

    # 加载权重
    if LoRA:
        model = load_model_with_lora(model, config, load_index)
    elif SFT:
        model = load_model_with_sft(model, config, load_index)
    elif Full:
        model = load_full_model(model, config, load_index, rank=gate_rank)
    else:
        model = load_pretrained_model(model, config, load_index)
        if model is None:
            print_processing("模型加载失败，无法启动交互")
            return

    print_processing("模型加载完成，进入交互式会话（输入 'quit' 退出，'clear' 清空历史）")

    conversation_history = []

    while True:
        user_input = input("\nUser：").strip()
        if user_input.lower() == 'quit':
            print_processing("退出会话")
            break
        if user_input.lower() == 'clear':
            conversation_history = []
            print_processing("会话历史已清空")
            continue

        history_text = ""
        for turn in conversation_history:
            if turn["role"] == "user":
                history_text += f"User：{turn['content']}"
            elif turn["role"] == "assistant":
                history_text += f"Assistant：{turn['content']}<|im_end|>"
        while len(Tokenizer.tokenize(history_text)) > max_len-256:
            history_text = history_text[30:]
        current_text = f"User：{user_input}Assistant："
        text = history_text + current_text


        seq_len = len(Tokenizer.tokenize(text))
        roll_len = len(Tokenizer.tokenize(history_text))
        ids = vocab.encode(text, max_len=max_len)

        if isinstance(ids, torch.Tensor):
            input_tensor = ids.long().to(config['device'])
        else:
            input_tensor = torch.tensor(ids, dtype=torch.long).to(config['device'])

        gate_ids = vocab.encode(current_text, max_len=max_len)
        if isinstance(gate_ids, torch.Tensor):
            gate_ids = gate_ids.long().to(config['device'])
        else:
            gate_ids = torch.tensor(gate_ids, dtype=torch.long).to(config['device'])

        model.eval()
        with torch.no_grad():
            generated_text,_ = vocab.generate(
                model=model,
                input_tensor=input_tensor,
                config=config,
                seq_len=seq_len,
                max_len=max_len,
                max_gen_len=512,
                repetition_penalty=repetition_penalty,
                FULL_MODEL=True,
                gate_input=gate_ids,
                roll_idx=roll_len,
                PRINT=True
            )

        out = "".join(generated_text)
        # 清理输出：去除前缀、特殊标记
        if out.startswith("Assistant："):
            out = out[len("Assistant："):]
        out = out.replace("<|im_end|>", "").strip()
        out = out.replace("[PAD]", "").strip()
        out = out.replace("User", "").strip()
        if "Assistant：" in out:
            out = out.rsplit("Assistant：", 1)[-1]

        if out:
            # print(f"\nAssistant：{out}")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": out})
        else:
            print("\nAssistant：[模型未生成有效回复]")


def start_chat_session(config_path, device, **kwargs):
    import json

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config['device'] = device
    default_params = {
        'vocab_data_path': 'dataset/minimind_dataset/pretrain_hq.jsonl',
        'vocab_trg_path': 'vocab.json',
        'max_len': 512,
        'repetition_penalty': 4,
        'load_index': 0,
        'LoRA': False,
        'SFT': False,
        'Full': True,
        'gate_rank': 64,
    }

    params = default_params.copy()
    params.update(kwargs)

    interactive_chat(config=config, **params)
