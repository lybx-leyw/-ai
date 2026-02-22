import sys
from torch.multiprocessing import freeze_support

def main():
    freeze_support()
    print("请选择要运行的训练脚本：")
    print("1. LoRA 训练 (lora_train.py)")
    print("2. 预训练 (pre_train.py)")
    print("3. 前缀训练 (prefix_train.py)")
    print("4. SFT 训练 (sft_train.py)")
    print("5. 门控训练 (train_gate.py)")
    choice = input("请输入数字 (1-5): ").strip()
    
    if choice == '1':from train_agent.lora_train import lora_train;lora_train()
    elif choice == '2':from train_agent.pre_train import pre_train;pre_train()
    elif choice == '3':from train_agent.prefix_train import prefix_train;prefix_train()
    elif choice == '4':from train_agent.sft_train import sft_train;sft_train()
    elif choice == '5':from train_agent.train_gate import train_gate;train_gate()
    else:print("无效输入，请输入1-5之间的数字。");sys.exit(1)

if __name__ == "__main__":
    main()