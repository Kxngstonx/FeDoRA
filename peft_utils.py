import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, lora_alpha, lora_dropout=0.0):
        super().__init__()
        self.r = r # rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r # rank가 작아지면 BA Output의 scale이 작아지는 문제 보정
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

class LoRALinear(LoRALayer):
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0, trainable_A: bool = False):
        super().__init__(linear_layer.in_features, linear_layer.out_features, r, lora_alpha, lora_dropout)

        self.linear = linear_layer
        self.linear.weight.requires_grad = False # Freeze W0
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False # Freeze bias

        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, linear_layer.in_features)), requires_grad=trainable_A)
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((linear_layer.out_features, r)))
        # Initial stat BA = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = self.linear(x) # W0 * x
        x_dropped = self.lora_dropout(x)
        result += (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling # BA * x → W0x + BAx
        return result

class DoRALinear(LoRALayer):
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0, trainable_A: bool = False):
        super().__init__(linear_layer.in_features, linear_layer.out_features, r, lora_alpha, lora_dropout)

        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, linear_layer.in_features)), requires_grad=trainable_A)
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((linear_layer.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=1, keepdim=True)) # Initialize: m = ||W0||_2

    def forward(self, x: torch.Tensor):
        W0 = self.linear.weight
        BA = (self.lora_B @ self.lora_A) * self.scaling
        V = W0 + BA

        V_norm = V.norm(p=2, dim=1, keepdim=True).detach() + 1e-8  # DoRA trick: stop gradient through norm
        W_new = self.m * (V / V_norm)

        x_dropped = self.lora_dropout(x)
        return F.linear(x_dropped, W_new, self.linear.bias)

class LoRAConv2d(LoRALayer):
    def __init__(self, conv_layer: nn.Conv2d, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0, trainable_A: bool = False):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, r, lora_alpha, lora_dropout)

        self.conv = conv_layer
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r, conv_layer.in_channels // conv_layer.groups, conv_layer.kernel_size[0], conv_layer.kernel_size[1])), requires_grad=trainable_A)
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
    def __init__(self, conv_layer: nn.Conv2d, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0, trainable_A: bool = False):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, r, lora_alpha, lora_dropout)

        self.conv = conv_layer
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r, conv_layer.in_channels // conv_layer.groups * conv_layer.kernel_size[0] * conv_layer.kernel_size[1])), requires_grad=trainable_A)
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
        V_norm = V.view(C_out, -1).norm(p=2, dim=1).view(C_out, 1, 1, 1).detach() + 1e-8  # DoRA trick: stop gradient through norm
        
        W_new = self.m * (V / V_norm)
        
        x_dropped = self.lora_dropout(x)
        return F.conv2d(x_dropped, W_new, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)

def inject_peft(model, peft_type="lora", r=8, lora_alpha=16, lora_dropout=0.0, trainable_A=False, skip_modules=None, global_skip_modules=None):
    if peft_type == "none":
        return model

    if skip_modules is None:
        skip_modules = []
    if global_skip_modules is None:
        global_skip_modules = []

    for name, module in model.named_children():
        if name in skip_modules or name in global_skip_modules:
            for param in module.parameters():
                param.requires_grad = True
            continue

        if len(list(module.children())) > 0:
            inject_peft(module, peft_type, r, lora_alpha, lora_dropout, trainable_A, global_skip_modules=global_skip_modules)

        if isinstance(module, nn.Linear):
            if peft_type == "lora":
                setattr(model, name, LoRALinear(module, r, lora_alpha, lora_dropout, trainable_A))
            elif peft_type == "dora":
                setattr(model, name, DoRALinear(module, r, lora_alpha, lora_dropout, trainable_A))
        elif isinstance(module, nn.Conv2d):
            if peft_type == "lora":
                setattr(model, name, LoRAConv2d(module, r, lora_alpha, lora_dropout, trainable_A))
            elif peft_type == "dora":
                setattr(model, name, DoRAConv2d(module, r, lora_alpha, lora_dropout, trainable_A))
    return model

def flex_lora_decompose_linear(W_global, W0, r, scaling):
    """
    W_global을 DoRA 파라미터 (m_new, B_new, A_new)로 SVD 분해.
    Alternating optimization 1 step:
      Step 1) Lambda_0 고정 → SVD로 B, A 구함
      Step 2) B, A 고정 → Lambda_1 업데이트

    Returns: (m_new, B_new, A_new, lambda_0, lambda_1)
      - m_new == lambda_0: shape (out, 1)
      - B_new: shape (out, r)
      - A_new: shape (r, in)
      - lambda_1: shape (out, 1), 검증/로깅용
    """
    # Lambda_0 = ||W_global||_row  (출력 뉴런별 norm)
    lambda_0 = W_global.norm(p=2, dim=1, keepdim=True)          # (out, 1)

    # Step 1: M_0 = W_global - W0, SVD로 rank-r 근사
    # 코드에서 BA_eff = B @ A * scaling 이므로 scaling 으로 나눈 뒤 SVD
    # bfloat16은 CUDA SVD 미지원이므로 float32로 캐스팅 후 복원
    orig_dtype = W_global.dtype
    M_0 = W_global - W0                                           # (out, in)
    U, S, Vh = torch.linalg.svd((M_0 / scaling).float(), full_matrices=False)
    U, S, Vh = U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
    actual_r = min(r, S.shape[0])
    B_new = U[:, :actual_r] * S[:actual_r].unsqueeze(0)          # (out, r)
    A_new = Vh[:actual_r, :]                                      # (r, in)

    # Truncation error: discarded singular values의 Frobenius norm × scaling (원래 행렬 스케일)
    S_total_norm = S.norm().item()
    trunc_err = (S[actual_r:].norm() * scaling).item() if actual_r < S.shape[0] else 0.0
    trunc_err_relative = (S[actual_r:].norm() / (S_total_norm + 1e-8)).item() if actual_r < S.shape[0] else 0.0

    # Step 2: Lambda_1 = ||(W0 + B_new @ A_new * scaling)||_row
    V_new = W0 + (B_new @ A_new) * scaling
    lambda_1 = V_new.norm(p=2, dim=1, keepdim=True)              # (out, 1)

    return lambda_0, B_new, A_new, lambda_0, lambda_1, trunc_err, trunc_err_relative, S


def flex_lora_decompose_conv2d(W_global, W0, r, scaling):
    """
    Conv2d W_global을 DoRA 파라미터 (m_new, B_new, A_new)로 SVD 분해.
    4D weight를 2D로 flatten해 처리하고 m shape만 복원.

    Returns: (m_new, B_new, A_new, lambda_0, lambda_1)
      - m_new == lambda_0: shape (C_out, 1, 1, 1)
      - B_new: shape (C_out, r)
      - A_new: shape (r, C_in_g * k1 * k2)
      - lambda_1: shape (C_out, 1, 1, 1), 검증/로깅용
    """
    C_out = W_global.shape[0]
    W_g2d = W_global.view(C_out, -1)
    W0_2d = W0.view(C_out, -1)

    lambda_0_2d = W_g2d.norm(p=2, dim=1, keepdim=True)           # (C_out, 1)
    lambda_0 = lambda_0_2d.view(C_out, 1, 1, 1)                  # (C_out, 1, 1, 1)

    orig_dtype = W_global.dtype
    M_0 = W_g2d - W0_2d
    U, S, Vh = torch.linalg.svd((M_0 / scaling).float(), full_matrices=False)
    U, S, Vh = U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
    actual_r = min(r, S.shape[0])
    B_new = U[:, :actual_r] * S[:actual_r].unsqueeze(0)          # (C_out, r)
    A_new = Vh[:actual_r, :]                                      # (r, flat_dim)

    S_total_norm = S.norm().item()
    trunc_err = (S[actual_r:].norm() * scaling).item() if actual_r < S.shape[0] else 0.0
    trunc_err_relative = (S[actual_r:].norm() / (S_total_norm + 1e-8)).item() if actual_r < S.shape[0] else 0.0

    V_new = W0_2d + (B_new @ A_new) * scaling
    lambda_1 = V_new.norm(p=2, dim=1, keepdim=True).view(C_out, 1, 1, 1)

    return lambda_0, B_new, A_new, lambda_0, lambda_1, trunc_err, trunc_err_relative, S


def flex_lora_decompose_linear_fixed_a(W_global, W0, A0, scaling):
    """
    A0 고정 시 Least Squares로 B_new만 계산. SVD 없이 O(r²) 역산.
    min_B ||（W_global - W0）/s - B @ A0||_F²  →  B_new = M @ A0† = solve(A0A0ᵀ, A0 @ Mᵀ)ᵀ
    """
    lambda_0 = W_global.norm(p=2, dim=1, keepdim=True)          # (out, 1)
    orig_dtype = W_global.dtype
    M_f = ((W_global - W0) / scaling).float()                    # (out, in)
    A0_f = A0.float()                                            # (r, in)
    AAt = A0_f @ A0_f.T                                          # (r, r)
    B_new = torch.linalg.solve(AAt, A0_f @ M_f.T).T.to(orig_dtype)  # (out, r)
    residual_norm = (M_f - B_new.float() @ A0_f).norm().item()
    ls_err = residual_norm * scaling
    ls_err_relative = residual_norm / (M_f.norm().item() + 1e-8)
    V_new = W0 + (B_new @ A0) * scaling
    lambda_1 = V_new.norm(p=2, dim=1, keepdim=True)
    return lambda_0, B_new, lambda_0, lambda_1, ls_err, ls_err_relative


def flex_lora_decompose_conv2d_fixed_a(W_global, W0, A0, scaling):
    """
    Conv2d 버전: 4D → 2D flatten 후 동일 Least Squares 적용.
    """
    C_out = W_global.shape[0]
    W_g2d = W_global.view(C_out, -1)
    W0_2d = W0.view(C_out, -1)
    lambda_0_2d = W_g2d.norm(p=2, dim=1, keepdim=True)
    lambda_0 = lambda_0_2d.view(C_out, 1, 1, 1)
    orig_dtype = W_global.dtype
    M_f = ((W_g2d - W0_2d) / scaling).float()
    A0_f = A0.float()
    AAt = A0_f @ A0_f.T
    B_new = torch.linalg.solve(AAt, A0_f @ M_f.T).T.to(orig_dtype)
    residual_norm = (M_f - B_new.float() @ A0_f).norm().item()
    ls_err = residual_norm * scaling
    ls_err_relative = residual_norm / (M_f.norm().item() + 1e-8)
    V_new = W0_2d + (B_new @ A0) * scaling
    lambda_1 = V_new.norm(p=2, dim=1, keepdim=True).view(C_out, 1, 1, 1)
    return lambda_0, B_new, lambda_0, lambda_1, ls_err, ls_err_relative


def flex_lora_aggregate(updated_client_weights, fed_avg_freqs, global_model, freeze_a=False):
    """
    FlexLoRA 서버 집계:
      1) 각 클라이언트의 (m_k, B_k, A_k)로 W_k = m_k * (V_k / ||V_k||_row) 재구성
      2) W_global = FedAvg(W_k)
      3a) freeze_a=False: SVD 분해 → m_new, B_new, A_new
      3b) freeze_a=True:  Least Squares → m_new, B_new (A_new = A₀ 고정)

    Returns:
      updated_dora_params: {"{layer_name}.m": tensor, ...}  ← global_model에 load할 파라미터
      lambda_logs: {layer_name: {'lambda_0', 'lambda_1', 'diff'}}  ← 로깅용
    """
    updated_dora_params = {}
    lambda_logs = {}

    for layer_name, module in global_model.named_modules():
        if not isinstance(module, (DoRALinear, DoRAConv2d)):
            continue
        is_conv = isinstance(module, DoRAConv2d)
        W0 = (module.conv.weight if is_conv else module.linear.weight).data
        r = module.r
        scaling = module.scaling

        # 클라이언트 루프: W_k 누적(방법 1)과 B/A/m 직접 평균 누적(방법 2)을 동시에 수행
        W_global = None   # 방법 1: FedAvg(W_k)
        B_avg = A_avg = m_avg = None  # 방법 2: FedAvg(B_k, A_k, m_k)
        for idx, (cid, sd) in enumerate(updated_client_weights.items()):
            freq = fed_avg_freqs[cid]
            A_k = sd[f"{layer_name}.lora_A"].to(W0.device)
            B_k = sd[f"{layer_name}.lora_B"].to(W0.device)
            m_k = sd[f"{layer_name}.m"].to(W0.device)

            if is_conv:
                C_out, C_in_g, k1, k2 = W0.shape
                BA_k = (B_k @ A_k).view(C_out, C_in_g, k1, k2) * scaling
                V_k = W0 + BA_k
                V_k_norm = V_k.view(C_out, -1).norm(p=2, dim=1).view(C_out, 1, 1, 1) + 1e-8
            else:
                BA_k = (B_k @ A_k) * scaling
                V_k = W0 + BA_k
                V_k_norm = V_k.norm(p=2, dim=1, keepdim=True) + 1e-8

            W_k = m_k * (V_k / V_k_norm)

            if idx == 0:
                W_global = W_k * freq
                B_avg = B_k * freq
                A_avg = A_k * freq
                m_avg = m_k * freq
            else:
                W_global += W_k * freq
                B_avg += B_k * freq
                A_avg += A_k * freq
                m_avg += m_k * freq

        # 방법 2: 평균낸 B, A, m으로 DoRA 공식 적용 → W_simple
        if is_conv:
            BA_avg = (B_avg @ A_avg).view(C_out, C_in_g, k1, k2) * scaling
            V_avg = W0 + BA_avg
            V_avg_norm = V_avg.view(C_out, -1).norm(p=2, dim=1).view(C_out, 1, 1, 1) + 1e-8
        else:
            BA_avg = (B_avg @ A_avg) * scaling
            V_avg = W0 + BA_avg
            V_avg_norm = V_avg.norm(p=2, dim=1, keepdim=True) + 1e-8
        W_simple = m_avg * (V_avg / V_avg_norm)

        # 방법 1 vs 방법 2 비교 통계
        W_diff = W_global - W_simple
        w_flex_norm = W_global.norm().item()
        w_diff = W_diff.norm().item()
        w_relative = w_diff / (w_flex_norm + 1e-8)

        # score layer : out_features < r인 경우 SVD로 rank-r 분해 불가 → FedAvg fallback
        max_svd_rank = min(W0.shape[0], W0.view(W0.shape[0], -1).shape[1])
        if max_svd_rank < r:
            updated_dora_params[f"{layer_name}.m"] = m_avg
            updated_dora_params[f"{layer_name}.lora_B"] = B_avg
            updated_dora_params[f"{layer_name}.lora_A"] = A_avg
            lambda_logs[layer_name] = {
                'lambda_0':             m_avg.detach().cpu(),
                'lambda_1':             m_avg.detach().cpu(),
                'singular_values':      torch.zeros(1),
                'lambda_diff':          0.0,
                'lambda_relative':      0.0,
                'w_diff':               w_diff,
                'w_relative':           w_relative,
                'trunc_err':            float('inf'),
                'trunc_err_relative':   float('inf'),
                'ls_err':               float('inf'),
                'ls_err_relative':      float('inf'),
            }
            continue

        if freeze_a:
            A0 = module.lora_A.data
            if is_conv:
                m_new, B_new, l0, l1, ls_err, ls_err_relative = flex_lora_decompose_conv2d_fixed_a(W_global, W0, A0, scaling)
            else:
                m_new, B_new, l0, l1, ls_err, ls_err_relative = flex_lora_decompose_linear_fixed_a(W_global, W0, A0, scaling)
            A_new = A0
            trunc_err = ls_err
            trunc_err_relative = 0.0
            sv = torch.zeros(1)
        elif is_conv:
            m_new, B_new, A_new, l0, l1, trunc_err, trunc_err_relative, sv = flex_lora_decompose_conv2d(W_global, W0, r, scaling)
            ls_err, ls_err_relative = 0.0, 0.0
        else:
            m_new, B_new, A_new, l0, l1, trunc_err, trunc_err_relative, sv = flex_lora_decompose_linear(W_global, W0, r, scaling)
            ls_err, ls_err_relative = 0.0, 0.0

        lambda_diff = (l1 - l0).norm().item()
        lambda_relative = lambda_diff / (l0.norm().item() + 1e-8)

        updated_dora_params[f"{layer_name}.m"] = m_new
        updated_dora_params[f"{layer_name}.lora_B"] = B_new
        updated_dora_params[f"{layer_name}.lora_A"] = A_new
        lambda_logs[layer_name] = {
            'lambda_0':             l0.detach().cpu(),
            'lambda_1':             l1.detach().cpu(),
            'singular_values':      sv.detach().cpu(),
            'lambda_diff':          lambda_diff,
            'lambda_relative':      lambda_relative,
            'w_diff':               w_diff,
            'w_relative':           w_relative,
            'trunc_err':            trunc_err,
            'trunc_err_relative':   trunc_err_relative,
            'ls_err':               ls_err,
            'ls_err_relative':      ls_err_relative,
        }

    return updated_dora_params, lambda_logs


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

def get_dora_delta_scalars(initial_components, current_components):
    """Returns (dm_per_layer, dv_per_layer) arrays, one scalar per layer per DoRA paper eq (3)(4)."""
    import numpy as np
    dm_list = []
    dv_list = []

    for name in initial_components.keys():
        if name in current_components:
            m0 = initial_components[name]['m'].view(-1)
            mt = current_components[name]['m'].view(-1)
            C_out = initial_components[name]['V'].shape[0]
            V0 = initial_components[name]['V'].view(C_out, -1)
            Vt = current_components[name]['V'].view(C_out, -1)
            dm_list.append(float(np.mean((mt - m0).abs().cpu().float().numpy())))
            cos_sim = F.cosine_similarity(V0, Vt, dim=1)
            dv_list.append(float(np.mean((1.0 - cos_sim).cpu().float().numpy())))

    return np.array(dm_list), np.array(dv_list)


def get_client_dora_delta_scalars(before_components, after_components):
    """Returns (dm_per_layer, dv_per_layer) arrays, one scalar per layer per DoRA paper eq (3)(4)."""
    import numpy as np
    dm_list = []
    dv_list = []

    for name in before_components.keys():
        if name not in after_components:
            continue
        m_before = before_components[name]['m'].view(-1)
        m_after  = after_components[name]['m'].view(-1)
        C_out = before_components[name]['V'].shape[0]
        V_before_flat = before_components[name]['V'].view(C_out, -1)
        V_after_flat  = after_components[name]['V'].view(C_out, -1)
        dm_list.append(float(np.mean((m_after - m_before).abs().cpu().float().numpy())))
        cos_sim = F.cosine_similarity(V_before_flat.float(), V_after_flat.float(), dim=1).clamp(-1.0, 1.0)
        dv_list.append(float(np.mean((1.0 - cos_sim).cpu().numpy())))

    return np.array(dm_list), np.array(dv_list)


def compute_temporal_dora_correlation(delta_m_series, delta_v_series):
    """Pearson correlation between ΔM and ΔD time series (paper's temporal method)."""
    import numpy as np
    dm, dv = np.array(delta_m_series), np.array(delta_v_series)
    if len(dm) < 2 or np.std(dm) == 0 or np.std(dv) == 0:
        return 0.0
    return float(np.corrcoef(dm, dv)[0, 1])

