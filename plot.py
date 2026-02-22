from tools import draw_plt
print("请选择过程：")
print("1. 预训练 (pre); 2. LoRA微调 (lora); 3. SFT微调 (sft)")

choice = input("请输入选项 (1-3): ").strip()
if choice == '1':process = 'pre';log = "logs\\log_pre"
elif choice == '2':process = 'lora';log = "logs\\log_lora"
elif choice == '3':process = 'gate';log = "log"
else:print("无效选择，默认选择预训练");process = 'pre';log = "logs\\log_pre"

fit_choice = input("是否使用对数拟合? (y/n，默认y): ").strip().lower()
use_fit = fit_choice != 'n'

draw_plt(process_name=process, log_name=log, use_log_fit=use_fit)
print(f"{process} 图表生成完成！")