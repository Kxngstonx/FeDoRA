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

class RAVANLinear(nn.Module):
    """Multi-head augmented LoRA from RAVAN (NeurIPS 2025).

    Each head i computes  s_i * B_i @ H_i @ A_i  where:
      - B_i (out, r) and A_i (r, in) are frozen at init
      - H_i (r, r) is the trainable core parameter
      - s_i (scalar) is a trainable scaling factor
    B_i / A_i are initialized with Gram-Schmidt to ensure orthogonal subspaces.
    """
    def __init__(self, linear_layer: nn.Linear, r: int = 110, lora_alpha: int = 110,
                 lora_dropout: float = 0.0, num_heads: int = 4,
                 init_method: str = 'gram_schmidt'):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.num_heads = num_heads
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # --- Initialize B_i and A_i with orthogonal subspaces ---
        if init_method == 'gram_schmidt':
            # B: Gram-Schmidt on concatenated columns
            B_concat = torch.randn(out_features, r * num_heads)
            Q_B, _ = torch.linalg.qr(B_concat)  # (out, r*h) orthonormal columns
            # A: Gram-Schmidt on concatenated rows (transpose, QR, transpose back)
            A_concat = torch.randn(r * num_heads, in_features)
            Q_A, _ = torch.linalg.qr(A_concat.T)  # (in, r*h)
            Q_A = Q_A.T  # (r*h, in) orthonormal rows
        else:  # random_normal — columns/rows are approximately orthogonal in high dim
            B_concat = torch.randn(out_features, r * num_heads)
            Q_B = B_concat
            A_concat = torch.randn(r * num_heads, in_features)
            Q_A = A_concat

        # Slice into per-head parameters (frozen)
        self.ravan_B = nn.ParameterList()
        self.ravan_A = nn.ParameterList()
        self.ravan_H = nn.ParameterList()
        self.ravan_s = nn.ParameterList()

        for i in range(num_heads):
            B_i = Q_B[:, i * r:(i + 1) * r].contiguous().clone()
            A_i = Q_A[i * r:(i + 1) * r, :].contiguous().clone()
            self.ravan_B.append(nn.Parameter(B_i, requires_grad=False))
            self.ravan_A.append(nn.Parameter(A_i, requires_grad=False))
            # H_i initialized to zero so ΔW starts at 0
            H_i = torch.zeros(r, r)
            self.ravan_H.append(nn.Parameter(H_i, requires_grad=True))
            # s_i initialized to 1
            s_i = torch.ones(1)
            self.ravan_s.append(nn.Parameter(s_i, requires_grad=True))

    def forward(self, x: torch.Tensor):
        result = self.linear(x)  # W0 * x
        x_dropped = self.lora_dropout(x)
        for i in range(self.num_heads):
            # s_i * B_i @ H_i @ A_i @ x * scaling
            delta = x_dropped @ self.ravan_A[i].T @ self.ravan_H[i].T @ self.ravan_B[i].T
            result = result + delta * (self.ravan_s[i] * self.scaling)
        return result


def inject_peft(model, peft_type="lora", r=8, lora_alpha=16, lora_dropout=0.0, trainable_A=False, skip_modules=None, global_skip_modules=None, target_modules=None, ravan_heads=4, ravan_init='gram_schmidt'):
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
            inject_peft(module, peft_type, r, lora_alpha, lora_dropout, trainable_A, global_skip_modules=global_skip_modules, target_modules=target_modules, ravan_heads=ravan_heads, ravan_init=ravan_init)

        # If target_modules is set, only inject into matching module names
        if target_modules is not None and name not in target_modules:
            continue

        if isinstance(module, nn.Linear):
            if peft_type == "lora":
                setattr(model, name, LoRALinear(module, r, lora_alpha, lora_dropout, trainable_A))
            elif peft_type == "dora":
                setattr(model, name, DoRALinear(module, r, lora_alpha, lora_dropout, trainable_A))
            elif peft_type == "ravan":
                setattr(model, name, RAVANLinear(module, r, lora_alpha, lora_dropout, num_heads=ravan_heads, init_method=ravan_init))
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
    A_new = Vh[:actual_r, :].clone()                             # (r, in)

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
    A_new = Vh[:actual_r, :].clone()                             # (r, flat_dim)

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


def flex_lora_decompose_linear_svd_a(W_global, W0, A_ref, scaling):
    """
    A_ref가 SVD로 추출된 공통 방향일 때 Least Squares로 B_new만 계산.
    """
    lambda_0 = W_global.norm(p=2, dim=1, keepdim=True)          # (out, 1)
    orig_dtype = W_global.dtype
    M_f = ((W_global - W0) / scaling).float()                    # (out, in)
    A0_f = A_ref.float()                                            # (r, in)
    AAt = A0_f @ A0_f.T                                          # (r, r)
    B_new = torch.linalg.solve(AAt, A0_f @ M_f.T).T.to(orig_dtype)  # (out, r)
    V_new = W0 + (B_new @ A_ref) * scaling
    lambda_1 = V_new.norm(p=2, dim=1, keepdim=True)
    
    B_f = B_new.float()
    M_recon_f = B_f @ A0_f
    ls_err = (M_f - M_recon_f).norm().item() * scaling
    M_norm = M_f.norm().item() * scaling
    ls_err_relative = ls_err / (M_norm + 1e-8)
    
    return lambda_0, B_new, lambda_0, lambda_1, ls_err, ls_err_relative

def flex_lora_decompose_conv2d_svd_a(W_global, W0, A_ref, scaling):
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
    A0_f = A_ref.float()
    AAt = A0_f @ A0_f.T
    B_new = torch.linalg.solve(AAt, A0_f @ M_f.T).T.to(orig_dtype)
    V_new = W0_2d + (B_new @ A_ref) * scaling
    lambda_1 = V_new.norm(p=2, dim=1, keepdim=True).view(C_out, 1, 1, 1)
    
    B_f = B_new.float()
    M_recon_f = B_f @ A0_f
    ls_err = (M_f - M_recon_f).norm().item() * scaling
    M_norm = M_f.norm().item() * scaling
    ls_err_relative = ls_err / (M_norm + 1e-8)
    
    return lambda_0, B_new, lambda_0, lambda_1, ls_err, ls_err_relative

def flex_lora_aggregate(updated_client_weights, fed_avg_freqs, global_model, freeze_a=False, svd_a=False):
    """
    FlexLoRA 서버 집계:
      1) 각 클라이언트의 (m_k, B_k, A_k)로 W_k = m_k * (V_k / ||V_k||_row) 재구성
      2) W_global = FedAvg(W_k)
      3a) freeze_a=False: SVD 분해 → m_new, B_new, A_new
      3b) freeze_a=True:  Least Squares → m_new, B_new (A_new = A₀ 고정)
      3c) svd_a=True:     A 행렬 SVD 추출 후 Least Squares → m_new, B_new (A_new = SVD(A_k))


    Returns:
      updated_dora_params: {"{layer_name}.m": tensor, ...}  ← global_model에 load할 파라미터
      lambda_logs: {layer_name: {'lambda_0', 'lambda_1', 'diff'}}  ← 로깅용
    """
    updated_dora_params = {}
    lambda_logs = {}

    for layer_name, module in global_model.named_modules():
        if not isinstance(module, (DoRALinear, DoRAConv2d, LoRALinear, LoRAConv2d)):
            continue
        
        is_conv = isinstance(module, (DoRAConv2d, LoRAConv2d))
        is_dora = isinstance(module, (DoRALinear, DoRAConv2d))
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
            m_k = sd[f"{layer_name}.m"].to(W0.device) if is_dora else None

            if is_conv:
                C_out, C_in_g, k1, k2 = W0.shape
                BA_k = (B_k @ A_k).view(C_out, C_in_g, k1, k2) * scaling
                V_k = W0 + BA_k
                if is_dora:
                    V_k_norm = V_k.view(C_out, -1).norm(p=2, dim=1).view(C_out, 1, 1, 1) + 1e-8
                    W_k = m_k * (V_k / V_k_norm)
                else:
                    W_k = V_k
            else:
                BA_k = (B_k @ A_k) * scaling
                V_k = W0 + BA_k
                if is_dora:
                    V_k_norm = V_k.norm(p=2, dim=1, keepdim=True) + 1e-8
                    W_k = m_k * (V_k / V_k_norm)
                else:
                    W_k = V_k


            if idx == 0:
                W_global = W_k * freq
                B_avg = B_k * freq
                A_avg = A_k * freq
                if is_dora: m_avg = m_k * freq
            else:
                W_global += W_k * freq
                B_avg += B_k * freq
                A_avg += A_k * freq
                if is_dora: m_avg += m_k * freq

        # 방법 2: 평균낸 B, A, m으로 DoRA 공식 적용 → W_simple
        if is_conv:
            BA_avg = (B_avg @ A_avg).view(C_out, C_in_g, k1, k2) * scaling
            V_avg = W0 + BA_avg
            if is_dora:
                V_avg_norm = V_avg.view(C_out, -1).norm(p=2, dim=1).view(C_out, 1, 1, 1) + 1e-8
                W_simple = m_avg * (V_avg / V_avg_norm)
            else:
                W_simple = V_avg
        else:
            BA_avg = (B_avg @ A_avg) * scaling
            V_avg = W0 + BA_avg
            if is_dora:
                V_avg_norm = V_avg.norm(p=2, dim=1, keepdim=True) + 1e-8
                W_simple = m_avg * (V_avg / V_avg_norm)
            else:
                W_simple = V_avg

        # 방법 1 vs 방법 2 비교 통계
        W_diff = W_global - W_simple
        w_flex_norm = W_global.norm().item()
        w_diff = W_diff.norm().item()
        w_relative = w_diff / (w_flex_norm + 1e-8)

        # score layer : out_features < r인 경우 SVD로 rank-r 분해 불가 → FedAvg fallback
        max_svd_rank = min(W0.shape[0], W0.view(W0.shape[0], -1).shape[1])
        if max_svd_rank < r:
            if is_dora:
                updated_dora_params[f"{layer_name}.m"] = m_avg
            updated_dora_params[f"{layer_name}.lora_B"] = B_avg
            updated_dora_params[f"{layer_name}.lora_A"] = A_avg
            lambda_logs[layer_name] = {
                'lambda_0':             m_avg.detach().cpu() if is_dora else torch.zeros(1),
                'lambda_1':             m_avg.detach().cpu() if is_dora else torch.zeros(1),
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
        elif svd_a:
            # A_k 행렬들을 수집하고 stack
            A_list = []
            for cid, sd in updated_client_weights.items():
                A_k = sd[f"{layer_name}.lora_A"].to(W0.device)
                if is_conv:
                    A_list.append(A_k.view(r, -1))
                else:
                    A_list.append(A_k)
            A_stack = torch.cat(A_list, dim=0) # (num_clients * r, in_features)

            # SVD를 통해 A_ref 추출
            U_A, S_A, Vh_A = torch.linalg.svd(A_stack.float(), full_matrices=False)
            actual_r = min(r, S_A.shape[0])
            A_ref = Vh_A[:actual_r, :].to(W0.dtype).clone()

            # B_new 계산 (Least Squares)
            if is_conv:
                # A_ref를 다시 원래 형태로 복원해야 함
                A_new = A_ref.view(r, C_in_g, k1, k2)
                m_new, B_new, l0, l1, ls_err, ls_err_relative = flex_lora_decompose_conv2d_svd_a(W_global, W0, A_ref, scaling)
            else:
                A_new = A_ref
                m_new, B_new, l0, l1, ls_err, ls_err_relative = flex_lora_decompose_linear_svd_a(W_global, W0, A_ref, scaling)

            trunc_err = ls_err # SVD_A는 A를 SVD로 추출하고 B는 LS로 구하므로 ls_err를 사용
            trunc_err_relative = 0.0
            sv = S_A # A_stack의 SVD 결과
        else:
            if is_conv:
                m_new, B_new, A_new, l0, l1, trunc_err, sv = flex_lora_decompose_conv2d(W_global, W0, r, scaling)
                ls_err, ls_err_relative = 0.0, 0.0
            else:
                m_new, B_new, A_new, l0, l1, trunc_err, trunc_err_relative, sv = flex_lora_decompose_linear(W_global, W0, r, scaling)
                ls_err, ls_err_relative = 0.0, 0.0

        lambda_diff = (l1 - l0).norm().item()
        lambda_relative = lambda_diff / (l0.norm().item() + 1e-8)

        if is_dora:
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


def fedex_lora_aggregate(updated_client_weights, fed_avg_freqs, global_model, r_prime=None):
    """
    FedEx-LoRA 서버 집계 알고리즘
    1) B_global = mean(B_i), A_global = mean(A_i)
    2) Residual = mean(B_i A_i) - B_global A_global
    3) Residual을 SVD로 저랭크(r_prime) 근사 후 복원 (통신 오버헤드 최적화 모사)
    4) 동결된 W0에 복원된 Residual을 영구적으로 더함 (Exact Aggregation)
    """
    updated_lora_params = {}
    fedex_logs = {}

    for layer_name, module in global_model.named_modules():
        # FedEx-LoRA는 본질적으로 LoRA의 A, B 행렬을 타겟으로 함
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue
        
        is_conv = isinstance(module, nn.Conv2d) or (hasattr(module, 'conv') and isinstance(module.conv, nn.Conv2d))
        W0 = module.conv.weight if is_conv else module.linear.weight
        r = module.r
        scaling = module.scaling
        actual_r_prime = r if r_prime is None else r_prime

        B_avg = A_avg = BA_sum = 0

        for idx, (cid, sd) in enumerate(updated_client_weights.items()):
            freq = fed_avg_freqs[cid]
            A_k = sd[f"{layer_name}.lora_A"].to(W0.device)
            B_k = sd[f"{layer_name}.lora_B"].to(W0.device)

            if is_conv:
                C_out, C_in_g, k1, k2 = W0.shape
                BA_k = (B_k @ A_k).view(C_out, C_in_g, k1, k2)
            else:
                BA_k = B_k @ A_k

            if idx == 0:
                B_avg = B_k * freq
                A_avg = A_k * freq
                BA_sum = BA_k * freq
            else:
                B_avg += B_k * freq
                A_avg += A_k * freq
                BA_sum += BA_k * freq

        # B_global A_global 행렬 곱 계산
        if is_conv:
            BA_avg = (B_avg @ A_avg).view(C_out, C_in_g, k1, k2)
        else:
            BA_avg = (B_avg @ A_avg)

        # 잔차 행렬(Residual Matrix) 계산: 스케일링(alpha/r) 반영 필수!
        Delta_W_res = (BA_sum - BA_avg) * scaling

        # 통신 오버헤드 최적화를 위한 Truncated SVD (Low-rank Approximation)
        M_res = Delta_W_res.view(W0.shape[0], -1).float()
        curr_r = min(actual_r_prime, min(M_res.shape[0], M_res.shape[1]))

        if curr_r > 0:
            U, S, Vh = torch.linalg.svd(M_res, full_matrices=False)
            M_rec = (U[:, :curr_r] @ torch.diag(S[:curr_r]) @ Vh[:curr_r, :]).to(W0.dtype)
        else:
            M_rec = torch.zeros_like(M_res, dtype=W0.dtype)

        if is_conv:
            Delta_W_rec = M_rec.view(C_out, C_in_g, k1, k2)
        else:
            Delta_W_rec = M_rec

        # W0의 그래디언트 차단 및 in-place 영구 업데이트 (메모리 누수 방지)
        with torch.no_grad():
            W0.data.add_(Delta_W_rec)

        updated_lora_params[f"{layer_name}.lora_B"] = B_avg
        updated_lora_params[f"{layer_name}.lora_A"] = A_avg
        fedex_logs[layer_name] = {'res_norm': Delta_W_res.norm().item(), 'rec_norm': Delta_W_rec.norm().item(), 'svd_r': curr_r}

    return updated_lora_params, fedex_logs


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


def ravan_aggregate(updated_client_weights, fed_avg_freqs, global_model):
    """RAVAN server aggregation (Algorithm 1, lines 16-18).

    For each RAVAN layer and each head i:
      1) Average the product  s_{c,i} * H_{c,i}  across participating clients
      2) Set the new global H_i = averaged product
      3) Reset s_i = 1 for the next round

    Returns:
      updated_params: dict of parameter name -> tensor to load into global_model
    """
    updated_params = {}

    for layer_name, module in global_model.named_modules():
        if not isinstance(module, RAVANLinear):
            continue

        num_heads = module.num_heads
        for i in range(num_heads):
            h_key = f"{layer_name}.ravan_H.{i}"
            s_key = f"{layer_name}.ravan_s.{i}"

            # Weighted average of s_i * H_i across clients
            sH_avg = None
            for idx, (cid, sd) in enumerate(updated_client_weights.items()):
                freq = fed_avg_freqs[cid]
                H_c = sd[h_key].to(module.ravan_H[i].device)
                s_c = sd[s_key].to(module.ravan_s[i].device)
                sH_c = s_c * H_c
                if idx == 0:
                    sH_avg = sH_c * freq
                else:
                    sH_avg = sH_avg + sH_c * freq

            # New H_i = averaged sH, s_i reset to 1
            updated_params[h_key] = sH_avg
            updated_params[s_key] = torch.ones(1, device=sH_avg.device)

    return updated_params
