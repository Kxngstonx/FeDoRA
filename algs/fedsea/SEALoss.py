import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import copy
import time
import numpy as np
# Assume other necessary imports like models, datasets are handled elsewhere

# --- FedSea Components ---

class GradientReversalFn(autograd.Function):
    """
    Gradient Reversal Layer Function
    Source: Equation 6 & 7 [cite: 29, 31]
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverses the gradient and scales it by lambda_ [cite: 31]
        output = grad_output.neg() * ctx.lambda_
        return output, None

def grad_reverse(x, lambda_=1.0):
    """ Wrapper for the Gradient Reversal Layer """
    return GradientReversalFn.apply(x, lambda_)

class IIDFeatureGenerator(nn.Module):
    """
    IID Feature Generator (Affine Transformation on masked features)
    Source: Equation 5 [cite: 21, 22]
    Applies learnable scale and bias only to features selected by the mask.
    """
    def __init__(self, feature_dim):
        super(IIDFeatureGenerator, self).__init__()
        # Learnable scale and bias for affine transformation
        self.scale = nn.Parameter(torch.ones(1, feature_dim))
        self.bias = nn.Parameter(torch.zeros(1, feature_dim))
        self.feature_dim = feature_dim

    def forward(self, features, mask):
        """
        Args:
            features (Tensor): Input features (batch_size, feature_dim)
            mask (Tensor): Binary mask (1 or 0) of shape (feature_dim,) or (1, feature_dim)
                           Indicates which features to apply affine transformation. [cite: 44]

        Returns:
            Tensor: Transformed features
        """
        if mask is None or mask.sum() == 0: # No mask or empty mask, return original
             return features

        # Ensure mask is broadcastable
        if mask.dim() == 1:
            mask = mask.unsqueeze(0) # Shape becomes (1, feature_dim)

        # Apply affine transformation only where mask is 1 [cite: 22]
        transformed_features = self.scale * features + self.bias
        
        # Select original features where mask is 0, transformed where mask is 1
        # Note: Equation 5 suggests concatenation, but this implementation applies transformation
        # in-place based on the mask, which is computationally simpler for dense features.
        # If features are distinctly separable/concatenated per mask, adjust accordingly.
        aligned_features = torch.where(mask.bool(), transformed_features, features)

        return aligned_features

class ClientDiscriminator(nn.Module):
    """
    Global Client Discriminator Network
    Source: Section IV-A-2[cite: 26], Equation 11 [cite: 54]
    Distinguishes which client the features came from.
    Input can be original or aligned features depending on the training step.
    """
    def __init__(self, feature_dim, num_clients):
        super(ClientDiscriminator, self).__init__()
        # Example: A simple MLP discriminator
        self.network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 2), # Output logits for each client
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 2), # Output logits for each client
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_clients), # Output logits for each client
            # No Softmax here, CrossEntropyLoss will handle it
        )
        self.feature_dim = feature_dim
        self.num_clients = num_clients

    def forward(self, features, attention_weights=None):
        """
        Args:
            features (Tensor): Input features (batch_size, feature_dim)
            attention_weights (Tensor, optional): Learned attention weights (alpha)
                                                 to scale features before discrimination.
                                                 Shape (feature_dim,) or (1, feature_dim). [cite: 45, 46]

        Returns:
            Tensor: Logits predicting the client index (batch_size, num_clients)
        """
        if attention_weights is not None:
             if attention_weights.dim() == 1:
                 attention_weights = attention_weights.unsqueeze(0) # Shape (1, feature_dim)
             # Weight features by attention [cite: 46]
             features = features * attention_weights

        client_logits = self.network(features)
        return client_logits

# --- Helper to get mask from attention ---
def get_mask_from_attention(attention_vector, k_ratio):
    """
    Generates a binary mask by selecting a proportion of features
    with the highest attention values.

    Args:
        attention_vector (Tensor): The learned attention vector alpha (feature_dim,).
        k_ratio (float): Proportion of features to select (between 0.0 and 1.0).
                         0.0 selects none, 1.0 selects all, 0.5 selects top 50%.

    Returns:
        Tensor: Binary mask (feature_dim,).
    """
    feature_dim = attention_vector.shape[0]

    k = int(round(feature_dim * k_ratio))
    k = max(1, min(feature_dim, k)) # Ensure k is within [1, feature_dim] for valid ratios (0, 1)

    if k == feature_dim: # If rounding results in selecting all features
         return torch.ones_like(attention_vector)

    kth_value_index = feature_dim - k + 1
    threshold = torch.kthvalue(attention_vector, kth_value_index).values

    mask = (attention_vector >= threshold).float()

    return mask