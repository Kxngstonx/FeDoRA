import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, lora_alpha, lora_dropout=0.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

class LoRALinear(LoRALayer):
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__(linear_layer.in_features, linear_layer.out_features, r, lora_alpha, lora_dropout)
        
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, linear_layer.in_features)), requires_grad=False)
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((linear_layer.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = self.linear(x)
        x_dropped = self.lora_dropout(x)
        result += (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result

class DoRALinear(LoRALayer):
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__(linear_layer.in_features, linear_layer.out_features, r, lora_alpha, lora_dropout)
        
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, linear_layer.in_features)), requires_grad=False)
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((linear_layer.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=1, keepdim=True))

    def forward(self, x: torch.Tensor):
        W0 = self.linear.weight
        BA = (self.lora_B @ self.lora_A) * self.scaling
        V = W0 + BA
        
        V_norm = V.norm(p=2, dim=1, keepdim=True) + 1e-8
        W_new = self.m * (V / V_norm)
        
        x_dropped = self.lora_dropout(x)
        return F.linear(x_dropped, W_new, self.linear.bias)

class LoRAConv2d(LoRALayer):
    def __init__(self, conv_layer: nn.Conv2d, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, r, lora_alpha, lora_dropout)
        
        self.conv = conv_layer
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r, conv_layer.in_channels // conv_layer.groups, conv_layer.kernel_size[0], conv_layer.kernel_size[1])), requires_grad=False)
        self.lora_B = nn.Parameter(self.conv.weight.new_zeros((conv_layer.out_channels, r, 1, 1)))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = self.conv(x)
        x_dropped = self.lora_dropout(x)
        lora_result = F.conv2d(x_dropped, self.lora_A, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        lora_result = F.conv2d(lora_result, self.lora_B)
        return result + lora_result * self.scaling

class DoRAConv2d(LoRALayer):
    def __init__(self, conv_layer: nn.Conv2d, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, r, lora_alpha, lora_dropout)
        
        self.conv = conv_layer
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r, conv_layer.in_channels // conv_layer.groups * conv_layer.kernel_size[0] * conv_layer.kernel_size[1])), requires_grad=False)
        self.lora_B = nn.Parameter(self.conv.weight.new_zeros((conv_layer.out_channels, r)))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        norm = self.conv.weight.view(self.conv.weight.shape[0], -1).norm(p=2, dim=1)
        self.m = nn.Parameter(norm.view(-1, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        W0 = self.conv.weight
        C_out, C_in_g, k1, k2 = W0.shape
        
        BA = (self.lora_B @ self.lora_A) * self.scaling
        BA = BA.view(C_out, C_in_g, k1, k2)
        
        V = W0 + BA
        V_norm = V.view(C_out, -1).norm(p=2, dim=1).view(C_out, 1, 1, 1) + 1e-8
        
        W_new = self.m * (V / V_norm)
        
        x_dropped = self.lora_dropout(x)
        return F.conv2d(x_dropped, W_new, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)

def inject_peft(model, peft_type="lora", r=8, lora_alpha=16, lora_dropout=0.0):
    if peft_type == "none":
        return model
    
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            inject_peft(module, peft_type, r, lora_alpha, lora_dropout)
            
        if isinstance(module, nn.Linear):
            if peft_type == "lora":
                setattr(model, name, LoRALinear(module, r, lora_alpha, lora_dropout))
            elif peft_type == "dora":
                setattr(model, name, DoRALinear(module, r, lora_alpha, lora_dropout))
        elif isinstance(module, nn.Conv2d):
            if peft_type == "lora":
                setattr(model, name, LoRAConv2d(module, r, lora_alpha, lora_dropout))
            elif peft_type == "dora":
                setattr(model, name, DoRAConv2d(module, r, lora_alpha, lora_dropout))
    return model

def get_dora_components(model):
    components = {}
    for name, module in model.named_modules():
        if isinstance(module, DoRAConv2d):
            W0 = module.conv.weight
            C_out, C_in_g, k1, k2 = W0.shape
            BA = (module.lora_B @ module.lora_A) * module.scaling
            BA = BA.view(C_out, C_in_g, k1, k2)
            V = W0 + BA
            components[name] = {'m': module.m.clone().detach(), 'V': V.clone().detach()}
        elif isinstance(module, DoRALinear):
            W0 = module.linear.weight
            BA = (module.lora_B @ module.lora_A) * module.scaling
            V = W0 + BA
            components[name] = {'m': module.m.clone().detach(), 'V': V.clone().detach()}
    return components

def calculate_dora_correlation(initial_components, current_components):
    import numpy as np
    delta_m_list = []
    delta_v_list = []
    
    for name in initial_components.keys():
        if name in current_components:
            m0 = initial_components[name]['m'].view(-1)
            mt = current_components[name]['m'].view(-1)
            
            V0 = initial_components[name]['V']
            Vt = current_components[name]['V']
            
            C_out = V0.shape[0]
            V0 = V0.view(C_out, -1)
            Vt = Vt.view(C_out, -1)
            
            # Change in magnitude
            delta_m = (mt - m0).abs()
            
            # Change in direction: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(V0, Vt, dim=1)
            delta_v = 1.0 - cos_sim
            
            delta_m_list.append(delta_m.cpu().numpy())
            delta_v_list.append(delta_v.cpu().numpy())
            
    if len(delta_m_list) == 0:
        return 0.0
        
    delta_m_all = np.concatenate(delta_m_list)
    delta_v_all = np.concatenate(delta_v_list)
    
    # Pearson correlation
    if len(delta_m_all) > 1 and np.std(delta_m_all) > 0 and np.std(delta_v_all) > 0:
        corr = np.corrcoef(delta_m_all, delta_v_all)[0, 1]
    else:
        corr = 0.0
        
    return corr

