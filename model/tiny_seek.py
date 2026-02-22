import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MLA,LayerNorm,MoE,TokenEmbedding

class ModelBlock(nn.Module):
    def __init__(
            self,d_model,n_head,d_c,d_r,device,hidden,
            other_experts,shared_experts,keep,
            dropout=0.1,ro_theta=10000.0,scale=0.02
        ):
        super().__init__()
        self.mla = MLA(d_model,d_c,d_r,n_head,device,ro_theta)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.moe_ffn = MoE(other_experts,shared_experts,d_model,hidden,device,keep,scale)
        self.drop = nn.Dropout(dropout)
    
    def forward(self,x,mask=None):
        attout = self.mla(x,x,mask)
        x = self.norm1(x+self.drop(attout))

        ffnout,lose = self.moe_ffn(x)
        x = self.norm2(x+self.drop(ffnout))
        return x, lose
    
class TinySeek(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.tokenEmbedding = TokenEmbedding(config['d_model'],config['vocab_size'])
        self.layers = nn.ModuleList(
            [
                ModelBlock(
                    config['d_model'],config['n_head'],config['d_c'],config['d_r'],
                    config['device'],config['hidden'],
                    config['other_experts'],config['shared_experts'],
                    config['keep'],
                    config['dropout'],config['ro_theta'],config['scale']
                 )
                for _ in range(config['n_layer'])
            ]
        )
        self.pad_idx = config['pad_idx']
        self.fc1 = nn.Linear(config['d_model'],config['vocab_size'])
        self.fc2 = nn.Linear(config['d_model'],config['vocab_size'])
        self.device = config['device']

    def get_mask(self,seq):
        _,seq_len = seq.shape
        casual_mask = torch.tril(torch.ones(seq_len,seq_len, device=self.device)).bool().unsqueeze(0).unsqueeze(0)
        pad_mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(3)
        pad_mask = pad_mask.repeat(1,1,1,seq_len)
        return casual_mask & pad_mask
    
    def forward(self, x):
        if x.dtype != torch.long:
            x = x.long() 

        mask = self.get_mask(x)
        x = self.tokenEmbedding(x)
        total_loss = 0.0
        for layer in self.layers:
            x, loss = layer(x,mask)
            total_loss += loss
        c_t = self.fc1(x)
        n_t = self.fc2(x)
        return c_t,n_t,total_loss

    
class gate_model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.tokenEmbedding = TokenEmbedding(config['d_model'],config['vocab_size'])
        self.layers = nn.ModuleList(
            [
                ModelBlock(
                    config['d_model'],config['n_head'],config['d_c'],config['d_r'],
                    config['device'],config['hidden'],
                    config['other_experts'],config['shared_experts'],
                    config['keep'],
                    config['dropout'],config['ro_theta'],config['scale']
                 )
                for _ in range(config['n_layer'])
            ]
        )
        self.pad_idx = config['pad_idx']
        self.fc1 = nn.Linear(config['d_model'],config['vocab_size'])
        self.fc2 = nn.Linear(config['d_model'],config['vocab_size'])
        self.device = config['device']

    def get_mask(self,seq):
        _,seq_len = seq.shape
        casual_mask = torch.tril(torch.ones(seq_len,seq_len, device=self.device)).bool().unsqueeze(0).unsqueeze(0)
        pad_mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(3)
        pad_mask = pad_mask.repeat(1,1,1,seq_len)
        return casual_mask & pad_mask
    
    def forward(self, x):
        if x.dtype != torch.long:
            x = x.long() 

        mask = self.get_mask(x)
        x = self.tokenEmbedding(x)
        for layer in self.layers:
            x,_ = layer(x,mask)

        return x
    
class gate(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.pad_idx = config['pad_idx']
        self.device = config['device']
        self.gate_model = gate_model(config=config)
        self.fc = nn.Linear(config['d_model'],6)
        self.dtype = self.fc.weight.dtype
    
    def get_token_value_positions_output(self, q_type, x, target_token_id):
        batch_size, seq_len = x.shape
        token_positions = []
        
        for i in range(batch_size):
            positions = torch.where(x[i] == target_token_id)[0]
            
            if len(positions) > 0:
                token_positions.append(positions[-1])
            else:
                non_pad_positions = torch.where(x[i] != self.pad_idx)[0]
                if len(non_pad_positions) > 0:
                    token_positions.append(non_pad_positions[-1])
                else:
                    token_positions.append(0) 
        
        token_positions = torch.tensor(token_positions, device=q_type.device)
        
        # 取对应位置的输出
        batch_indices = torch.arange(batch_size, device=q_type.device)
        specific_output = q_type[batch_indices, token_positions, :]  # [batch, 6]
        
        return specific_output

    def forward(self, x, target_token_id=10):
        if x.dtype != torch.long:
            x = x.long() 
        
        gate_x = self.gate_model(x)
        q_type = self.fc(gate_x)  # [batch, seq_len, 6]
        q_type = self.get_token_value_positions_output(q_type, x, target_token_id)
        
        return q_type

class gated_TinySeek(nn.Module):
    def __init__(self, seq_len, config, rank=64):
        super().__init__()
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.seq_len = seq_len
        self.model = nn.ModuleList([
            TinySeek(config=config) for _ in range(6)
        ])
        self.gate = gate(config=config) 
        from model.lora_model import apply_lora
        apply_lora(model=self.gate.gate_model, rank=rank, scale=0.02)
    
    def load_part_model(self, model_1, model_2, model_3, model_4, model_5, model_6, gate_model, rank=64, prefix_1=False):
        models = [model_1, model_2, model_3, model_4, model_5, model_6] 
        for i, trained_model in enumerate(models):
            if i < len(self.model):
                try:
                    self.model[i].load_state_dict(trained_model.state_dict())
                except:
                    from model.lora_model import apply_lora
                    if i == 0:prefix = prefix_1
                    else: prefix = True
                    apply_lora(model=self.model[i], rank=rank, scale=0.02, prefix=prefix)
                    self.model[i].load_state_dict(trained_model.state_dict())
        self.gate.gate_model.load_state_dict(gate_model.state_dict())
        
        for trained_model in self.model:
            for param in trained_model.parameters():
                param.requires_grad = False
        for param in self.gate.gate_model.parameters():
            param.requires_grad = False
        for name, module in self.gate.gate_model.named_modules():
            if 'lora' in name.lower() or 'prefix' in name.lower():  
                for param in module.parameters():
                    param.requires_grad = True
                print(f"保留Gate_LoRA层: {name}")
    
    def load_part_model_only(self, model_1, model_2, model_3, model_4, model_5, model_6, rank=64, prefix_1=False):
        models = [model_1, model_2, model_3, model_4, model_5, model_6] 
        for i, trained_model in enumerate(models):
            if i < len(self.model):
                try:
                    self.model[i].load_state_dict(trained_model.state_dict())
                except:
                    from model.lora_model import apply_lora
                    if i == 0:prefix = prefix_1
                    else: prefix = True
                    apply_lora(model=self.model[i], rank=rank, scale=0.02, prefix=prefix)
                    self.model[i].load_state_dict(trained_model.state_dict())

    def forward(self, x, gate_input=None, roll_idx=0):  # [batch, seq_len]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, seq_len = x.shape
        device = x.device
        x_long = x if x.dtype == torch.long else x.long().to(device)
        if gate_input is None:
            gate_input = x_long
        else:
            gate_input = gate_input if gate_input.dtype == torch.long else gate_input.long().to(device)
        G_x = self.gate(gate_input)  
        G_x = torch.softmax(G_x, dim=-1)  # [batch, num_experts]

        expert_assignments = torch.argmax(G_x, dim=-1)  # [batch]
        # print(expert_assignments.clone().tolist())
        
        output_1 = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        output_2 = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        
        with torch.no_grad():
            for expert_idx, expert in enumerate(self.model):
                mask = (expert_assignments == expert_idx)  # [batch]
                
                if not mask.any():  
                    continue
                
                if batch_size == 1:
                    if mask.item():
                        if expert_assignments.item() == 1 or expert_assignments.item() == 3 or \
                            expert_assignments.item() == 4:
                            if x_long[0,roll_idx] != 9:
                                import numpy as np
                                user_indices = np.where(x_long[0] == 9)[0]
                                if user_indices.size == 0:
                                    raise ValueError("错误，输入里没有\'User\'")
                                roll_idx = user_indices[-1]
                            model_output_1, model_output_2, _ = expert(gate_input)
                            output_1 = torch.roll(model_output_1.float(), shifts=roll_idx, dims=-2)
                            output_2 = torch.roll(model_output_2.float(), shifts=roll_idx, dims=-2)
                        else:
                            model_output_1, model_output_2, _ = expert(x_long)
                            output_1 = model_output_1.float()
                            output_2 = model_output_2.float()
                else:
                    x_input = x_long[mask]
                    model_output_1, model_output_2, _ = expert(x_input)
                    output_1[mask] = model_output_1.float()
                    output_2[mask] = model_output_2.float()
        
        return output_1, output_2, 0
    
    def cacu_gate_loss(self, batch_trg_1, batch_trg_2, config, x):
        batch_size = x.shape[0]
        device = x.device
        
        if x.dtype != self.gate.dtype:
            x = x.to(self.gate.dtype)
        gate_output = self.gate(x)  # [batch_size, 6]
        model_losses = torch.zeros(batch_size, 6).to(device)
        
        with torch.no_grad():
            batch_size = x.size(0)
            num_experts = len(self.model)
            model_losses = torch.zeros(batch_size, num_experts, device=device, dtype=torch.float32)

            seq_len = batch_trg_1.size(1)
            pad_mask = (batch_trg_1 != config['pad_idx']).view(batch_size, seq_len)
            valid_counts = pad_mask.sum(dim=1).clamp(min=1).to(device)

            for expert_idx, expert in enumerate(self.model):
                expert_training = expert.training
                expert.eval()

                model_output_1, model_output_2, _ = expert(x)

                loss_1_per_token = F.cross_entropy(
                    model_output_1.reshape(-1, self.vocab_size),
                    batch_trg_1.reshape(-1),
                    ignore_index=config['pad_idx'],
                    reduction='none'
                ).reshape(batch_size, seq_len)

                loss_2_per_token = F.cross_entropy(
                    model_output_2.reshape(-1, self.vocab_size),
                    batch_trg_2.reshape(-1),
                    ignore_index=config['pad_idx'],
                    reduction='none'
                ).reshape(batch_size, seq_len)

                mask1 = (batch_trg_1 != config['pad_idx']).to(loss_1_per_token.dtype)
                mask2 = (batch_trg_2 != config['pad_idx']).to(loss_2_per_token.dtype)
                valid1 = mask1.sum(dim=1).clamp(min=1).to(device)
                valid2 = mask2.sum(dim=1).clamp(min=1).to(device)

                mean1 = (loss_1_per_token * mask1).sum(dim=1) / valid1
                mean2 = (loss_2_per_token * mask2).sum(dim=1) / valid2

                loss_per_sample = mean1 + mean2

                model_losses[:, expert_idx] = loss_per_sample 

                # 恢复专家原始训练模式
                if expert_training:
                    expert.train()

        best_model_indices = torch.argmin(model_losses, dim=-1)  # [batch_size]
        print(best_model_indices.clone().tolist())
                
        # 计算gate的交叉熵损失
        gate_loss = F.cross_entropy(gate_output, best_model_indices)
        gate_predictions = gate_output.argmax(dim=1)
        print(gate_predictions.clone().tolist())
        correct = (gate_predictions == best_model_indices).sum().item()
        total = best_model_indices.size(0)
        gate_accuracy = correct / total

        return gate_loss,gate_accuracy
