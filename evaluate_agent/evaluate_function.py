from model.tiny_seek import TinySeek
from model.lora_model import apply_lora
from tools import Vocab
import torch
import torch.nn.functional as F
from tools import print_processing
from tools import Tokenizer

def load_model_with_lora(model, config, load_index):
    apply_lora(model=model, rank=64, scale=0.02)
    filepath = f"out\\TinySeek_Lora{load_index}.pkl"
    try:
        state_dict = torch.load(filepath, map_location=config['device'])
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
            print(f"成功加载模型状态字典: {filepath}")
        else:
            # 如果是完整模型，提取其状态字典
            loaded_model = state_dict
            model.load_state_dict(loaded_model.state_dict(), strict=False)
            print(f"成功从完整模型加载参数: {filepath}")
    except FileNotFoundError:
        print(f"未找到模型参数: {filepath}")
    except Exception as e:
        print(f"加载模型参数时出错: {e}")
    return model.to(config['device'])

def load_model_with_sft(model, config, load_index):
    filepath = f"out\\TinySeek_SFT{load_index}.pkl"
    try:
        # 先尝试加载状态字典
        state_dict = torch.load(filepath, map_location=config['device'])
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
            print(f"成功加载模型状态字典: {filepath}")
        else:
            # 如果是完整模型，提取其状态字典
            loaded_model = state_dict
            model.load_state_dict(loaded_model.state_dict(), strict=False)
            print(f"成功从完整模型加载参数: {filepath}")
    except FileNotFoundError:
        print(f"未找到模型参数: {filepath}")
    except Exception as e:
        print(f"加载模型参数时出错: {e}")
    return model.to(config['device'])

def load_full_model(model, config, load_index, rank=64):
    filepath = f"out\\TinySeek_Full{load_index}.pkl"
    apply_lora(model=model.gate.gate_model, rank=rank, scale=0.02)
    model_1 = TinySeek(config).to(config['device'])
    model_2 = TinySeek(config).to(config['device'])
    model_3 = TinySeek(config).to(config['device'])
    model_4 = TinySeek(config).to(config['device'])
    model_5 = TinySeek(config).to(config['device'])
    model_6 = TinySeek(config).to(config['device'])
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
    model.load_part_model_only(
        model_1=model_1,model_2=model_2,model_3=model_3,
        model_4=model_4,model_5=model_5,model_6=model_6
        )
    def init_load(model,filepath,config):
        # 先尝试加载状态字典
        state_dict = torch.load(filepath, map_location=config['device'])
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
            print(f"成功加载模型状态字典: {filepath}")
        else:
            # 如果是完整模型，提取其状态字典
            loaded_model = state_dict
            model.load_state_dict(loaded_model.state_dict(), strict=False)
            print(f"成功从完整模型加载参数: {filepath}")
    try:
        init_load(model,filepath,config)
    except:
        try:
            filepath = f"out\\TinySeek_gate{load_index}.pkl"
            init_load(model,filepath,config)
        except FileNotFoundError:
            print(f"未找到模型参数: {filepath}")
        except Exception as e:
            print(f"加载模型参数时出错: {e}")
    return model.to(config['device'])

def load_pretrained_model(model, config, load_index):
    filepath = f"out\\TinySeek_Pre{load_index}.pkl"
    try:
        # 加载模型参数
        state_dict = torch.load(filepath, map_location=config['device'])
        
        # 如果是完整模型，提取其状态字典
        if not isinstance(state_dict, dict):
            state_dict = state_dict.state_dict()
        
        # 适配词表大小
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('fc1') or key.startswith('fc2') or key.startswith('tokenEmbedding'):
                # 根据词表大小截取或保留
                if value.size(0) > config['vocab_size']:
                    new_state_dict[key] = value[:config['vocab_size']]
                elif value.size(0) < config['vocab_size']:
                    new_state_dict[key] = value  
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        
        # 加载适配后的状态字典
        model.load_state_dict(new_state_dict, strict=False)
        print(f"成功加载适配后的模型参数：{filepath}")
        return model.to(config['device'])
        
    except FileNotFoundError:
        print(f"未找到模型参数：{filepath},请先完成训练再进行评估")
        return None
    except Exception as e:
        print(f"加载模型参数时出错：{e}")
        return None
    
def get_topk_predictions(output_tensor, k=5):
    probs = F.softmax(output_tensor, dim=-1)
    topk = torch.topk(probs, k=k, dim=-1)
    return topk.indices.squeeze(0).long().tolist(), topk.values.squeeze(0).tolist()

def print_topk_results(top5_1_indices, top5_1_vals, top5_2_indices, top5_2_vals, vocab):
    print("位置\t预测词1\t对应概率\t预测词2\t对应概率")
    for i in range(len(top5_1_indices)):
        print(f"{i}\t"
              f"{vocab.decode(top5_1_indices[i:i+1])}\t{top5_1_vals[i]:4.2f}\t"
              f"{vocab.decode(top5_2_indices[i:i+1])}\t{top5_2_vals[i]:4.2f}")

def calculate_perplexity(model, input_tensor, config):
    with torch.no_grad():
        c_t, n_t, load_loss = model(input_tensor)
        labels_1 = torch.roll(input_tensor[0], shifts=-1, dims=-1)
        labels_1[-1] = config['pad_idx']
        labels_2 = torch.roll(labels_1, shifts=-1, dims=-1)
        labels_2[-1] = config['pad_idx']
        
        cross_loss_1 = F.cross_entropy(c_t.view(-1, c_t.size(-1)), labels_1.view(-1), ignore_index=config['pad_idx'])
        cross_loss_2 = F.cross_entropy(n_t.view(-1, n_t.size(-1)), labels_2.view(-1), ignore_index=config['pad_idx'])
        
        perplexity_1 = torch.exp(cross_loss_1) 
        perplexity_2 = torch.exp(cross_loss_2)
        
        return perplexity_1.item(), perplexity_2.item(), load_loss

def evaluate_model(
        config, vocab_data_path, vocab_trg_path, max_len, repetition_penalty, load_index=0, LoRA=False, SFT=False,
        Full=False, gate_rank=64
        ): 
    vocab = Vocab(vocab_data_path, vocab_trg_path, config['vocab_size']) 
    if Full:
        from model.tiny_seek import gated_TinySeek
        model = gated_TinySeek(seq_len=max_len,config=config)
    else:
        model = TinySeek(config).to(config['device'])
    
    if LoRA == True:
        model = load_model_with_lora(model, config, load_index)
    elif SFT == True:
        model = load_model_with_sft(model, config, load_index)
    elif Full == True:
        model = load_full_model(model, config, load_index, rank=gate_rank)
    else:
        model = load_pretrained_model(model, config, load_index)
        if model is None:
            return
    
    print_processing("模型加载完成，开始评估...")
    text = input("请输入评估文本：")
    if LoRA == True or SFT == True or Full == True:
        text = "User：" + text + "Assistant："
    ids = vocab.encode(text, max_len=max_len)
    input_tensor = ids
    model.eval()
    
    with torch.no_grad():
        print_processing("第一项评估：词表概率分布")
        c_t, n_t, _ = model(input_tensor)

        seq_len = len(Tokenizer.tokenize(text))
        c_t = c_t[:, seq_len-1, :]
        n_t = n_t[:, seq_len-1, :]

        top5_1_indices, top5_1_vals = get_topk_predictions(c_t, k=5)
        top5_2_indices, top5_2_vals = get_topk_predictions(n_t, k=5)
        
        print_topk_results(top5_1_indices, top5_1_vals, top5_2_indices, top5_2_vals, vocab)

        print_processing("第二项评估：自回归生成文本")
        max_gen_len = int(input("请输入生成文本的最大长度（建议不超过20）:"))

        generated_text, keep_cnt = vocab.generate(
            model=model,
            input_tensor=input_tensor,
            config=config,
            seq_len=seq_len,
            max_len=max_len,
            max_gen_len=max_gen_len,
            repetition_penalty=repetition_penalty
        )
        
        print("生成文本：", "".join(generated_text))
        print(f"保留头2预测结果的次数：{keep_cnt}")
        
        if LoRA == False and SFT == False and Full == False:
            print_processing("第三项评估：输入完整文本，计算模型两个头困惑度")
            text = input("请输入完整评估文本(建议长一些，以测试模型的长文本处理能力):")
            ids = vocab.encode(text, max_len=max_len)
            input_tensor = ids
            model.eval()
            
            perplexity_1, perplexity_2, load_loss = calculate_perplexity(model, input_tensor, config)
            print(f"困惑度头1: {perplexity_1:.2f}, 困惑度头2: {perplexity_2:.2f}, 负载均衡损失: {load_loss}")
            
            print_processing("第四项评估：评估长文本的两头困惑度")
            long_text = text * 10
            ids = vocab.encode(long_text, max_len=10 * max_len)
            input_tensor = ids
            model.eval()
            
            perplexity_1, perplexity_2, load_loss = calculate_perplexity(model, input_tensor, config)
            print(f"长文本困惑度头1: {perplexity_1:.2f}, 长文本困惑度头2: {perplexity_2:.2f}, 负载均衡损失: {load_loss}")

        print_processing("评估完成！")