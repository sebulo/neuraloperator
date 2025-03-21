from typing import List, Optional, Tuple, Union

from ..utils import validate_scaling_factor

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .resample import resample

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def get_fp16_spacing(x: torch.Tensor):
    """
    Calculate the spacing (ULP) for FP16 numbers.
    For FP16, the spacing is 2^(e-10) where e is the exponent.
    """
    # Convert to FP32 for manipulation
    x = x.float()
    # Get absolute value
    abs_x = torch.abs(x)
    # Handle zero specially
    spacing = torch.where(abs_x == 0,
                         torch.tensor(2**-24, device=x.device),  # minimum positive FP16
                         abs_x * 2**-10)  # normal spacing
    return spacing

def stochastic_round_fp16(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an FP32 tensor to FP16 using stochastic rounding.
    The FP32 tensor is first converted deterministically to FP16 (and back to FP32)
    to obtain the candidate value. The spacing (ULP) at the candidate is computed,
    and the remainder is used to decide probabilistically whether to round up.
    """
    candidate = x.half().float()         # Deterministic FP16 conversion (FP32 representation)
    spacing = get_fp16_spacing(candidate)  # Get the ULP (spacing between representable FP16 numbers)
    remainder = x - candidate
    prob = (remainder / spacing).clamp(0, 1) # Compute probability to round up
    rand = torch.rand(x.shape, device=x.device)
    round_up = (rand < prob).to(torch.float32)
    result = candidate + spacing * round_up
    return result

def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    
    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors. 
        if False, 
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]


class SpectralConv(BaseSpectralConv):
    def __init__(self, in_channels, out_channels, n_modes, complex_data=False, max_n_modes=None, bias=True, 
                 separable=False, resolution_scaling_factor=None, fno_block_precision="full", 
                 rank=0.5, factorization=None, implementation="reconstructed", fixed_rank_modes=False, 
                 decomposition_kwargs=None, init_std="auto", fft_norm="forward", device=None):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_data = complex_data
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self.max_n_modes = max_n_modes if max_n_modes is not None else self.n_modes
        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.resolution_scaling_factor = validate_scaling_factor(resolution_scaling_factor, self.order)
        self.separable = separable
        self.fft_norm = fft_norm

        # Ensure weight_shape matches input channels
        weight_shape = (in_channels, *self.max_n_modes) if separable else (in_channels, out_channels, *self.max_n_modes)
        self.weight_dtype = {"full": torch.cfloat, "half": torch.chalf, "mixed": torch.chalf}.get(fno_block_precision, torch.cfloat)
        init_std = (2 / (in_channels + out_channels))**0.5 if init_std == "auto" else init_std
        
        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        if factorization == 'tucker':
            self.weight = FactorizedTensor.new(weight_shape, rank=self.rank, factorization=factorization, 
                                             fixed_rank_modes=fixed_rank_modes, **tensor_kwargs, dtype=self.weight_dtype)
            self.weight.normal_(0, init_std)
        else:
            self.weight = torch.empty(weight_shape, dtype=self.weight_dtype, device=device)
            self.weight.normal_(0, init_std)
        
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, *(1,) * self.order)) if bias else None
        self.mask = None  # Set by optimizer for sparse backward

    def forward(self, x, output_shape=None):
        batchsize, channels, *mode_sizes = x.shape
        device = x.device
        
        # Verify input channels match weight
        if channels != self.in_channels:
            raise ValueError(f"Input channels {channels} do not match expected {self.in_channels}")

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        # Dense forward with sparse backward
        out = sparse_gradient_spectral_conv(x, self.weight, self.mask, self.bias, fft_size, fft_dims, 
                                          self.complex_data, self.fft_norm, self.separable)
        
        # For mixed precision, convert output back to full precision
        if self.fno_block_precision == "mixed" and out.dtype == torch.complex32:
            out_real = stochastic_round_fp16(out.real)
            out_imag = stochastic_round_fp16(out.imag)
            out = torch.complex(out_real, out_imag)
        
        if output_shape:
            out = resample(out, 1.0, list(range(2, out.ndim)), output_shape=output_shape)
        return out

    def set_optimizer_mask(self, mask):
        """Set the sparsity mask from the optimizer for backward pass."""
        self.mask = mask
    
    def transform(self, x, output_shape=None):
        """Transform input to match desired output shape.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        output_shape : tuple, optional
            Desired output spatial dimensions, by default None
            If None, no resampling is performed
            
        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        if output_shape is None:
            return x
        
        return resample(x, 1.0, list(range(2, x.ndim)), output_shape=output_shape)


import torch
from torch.autograd import Function
from tensorly import tenalg

class SparseGradientSpectralConv(Function):
    @staticmethod
    def forward(ctx, x, weight, mask, bias, fft_size, fft_dims, complex_data, fft_norm, separable):
        """Dense forward pass with spectral convolution."""
        device = x.device
        batchsize, channels = x.shape[:2]
        
        # Ensure all tensors are on the same device
        weight = weight.to(device)
        if mask is not None:
            mask = mask.to(device)
        if bias is not None:
            bias = bias.to(device)

        ctx.save_for_backward(x, weight, bias)
        ctx.mask = mask
        ctx.fft_size = fft_size
        ctx.fft_dims = fft_dims
        ctx.complex_data = complex_data
        ctx.fft_norm = fft_norm
        ctx.separable = separable

        # Compute FFT
        if complex_data:
            x_fft = torch.fft.fftn(x, norm=fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x_fft = torch.fft.rfftn(x, norm=fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]

        if len(dims_to_fft_shift) > 0:
            x_fft = torch.fft.fftshift(x_fft, dim=dims_to_fft_shift)

        out_channels = weight.shape[1] if not separable else channels
        out_fft = torch.zeros([batchsize, out_channels, *fft_size], 
                            dtype=weight.dtype, device=device)

        # Get the actual sizes for each dimension from the weight tensor
        if separable:
            weight_spatial_dims = weight.shape[1:]  # ci...
            out_fft_slices = [slice(None), slice(None)]  # b,c
            weight_slices = [slice(None)]  # c
        else:
            weight_spatial_dims = weight.shape[2:]  # cio...
            out_fft_slices = [slice(None), slice(None)]  # b,c
            weight_slices = [slice(None), slice(None)]  # c,o
            
        # Add spatial dimension slices based on weight size
        for dim_size in weight_spatial_dims:
            out_fft_slices.append(slice(0, dim_size))
            weight_slices.append(slice(0, dim_size))
            
        # Extract relevant part of FFT
        x_fft_subset = x_fft[out_fft_slices]
        weight_subset = weight[weight_slices]
        
        # Perform einsum with correctly sized tensors
        if separable:
            out_fft[out_fft_slices] = torch.einsum('bci...,ci...->bc...', x_fft_subset, weight_subset)
        else:
            out_fft[out_fft_slices] = torch.einsum('bci...,cio...->bco...', x_fft_subset, weight_subset)

        if len(dims_to_fft_shift) > 0:
            out_fft = torch.fft.fftshift(out_fft, dim=dims_to_fft_shift)

        if complex_data:
            out = torch.fft.ifftn(out_fft, s=x.shape[2:], dim=fft_dims, norm=fft_norm)
        else:
            out = torch.fft.irfftn(out_fft, s=x.shape[2:], dim=fft_dims, norm=fft_norm)
        
        # Apply stochastic rounding for complex32 (half precision complex) outputs
        if out.dtype == torch.complex32:
            out_real = stochastic_round_fp16(out.real)
            out_imag = stochastic_round_fp16(out.imag)
            out = torch.complex(out_real, out_imag)
        
        if bias is not None:
            out = out + bias

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Sparse backward pass computing gradients only for active dimensions."""
        x, weight, bias = ctx.saved_tensors
        mask = ctx.mask  # This should indicate which dimensions are active
        fft_size = ctx.fft_size
        fft_dims = ctx.fft_dims
        complex_data = ctx.complex_data
        fft_norm = ctx.fft_norm
        separable = ctx.separable

        grad_x = grad_weight = grad_bias = None

        # Get FFT of grad_output
        if complex_data:
            grad_fft = torch.fft.fftn(grad_output, norm=fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            grad_fft = torch.fft.rfftn(grad_output, norm=fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]

        if len(dims_to_fft_shift) > 0:
            grad_fft = torch.fft.fftshift(grad_fft, dim=dims_to_fft_shift)

        # Get FFT of input
        if complex_data:
            x_fft = torch.fft.fftn(x, norm=fft_norm, dim=fft_dims)
        else:
            x_fft = torch.fft.rfftn(x, norm=fft_norm, dim=fft_dims)

        if len(dims_to_fft_shift) > 0:
            x_fft = torch.fft.fftshift(x_fft, dim=dims_to_fft_shift)

        # Get the actual sizes for each dimension from the weight tensor
        if separable:
            weight_spatial_dims = weight.shape[1:]  # ci...
            out_fft_slices = [slice(None), slice(None)]  # b,c
            weight_slices = [slice(None)]  # c
        else:
            weight_spatial_dims = weight.shape[2:]  # cio...
            out_fft_slices = [slice(None), slice(None)]  # b,c
            weight_slices = [slice(None), slice(None)]  # c,o
            
        # Add spatial dimension slices based on weight size
        for dim_size in weight_spatial_dims:
            out_fft_slices.append(slice(0, dim_size))
            weight_slices.append(slice(0, dim_size))

        # Apply mask to weight if provided
        if mask is not None:
            breakpoint()
            weight = weight * mask
            

        # Gradient w.r.t. input (x)
        if ctx.needs_input_grad[0]:
            grad_x_fft = torch.zeros_like(x_fft)
            x_fft_subset = grad_fft[out_fft_slices]
            weight_subset = weight[weight_slices]
            
            if separable:
                grad_x_fft[out_fft_slices] = torch.einsum('bc...,ci...->bci...', 
                    x_fft_subset, 
                    weight_subset.conj() if complex_data else weight_subset)
            else:
                grad_x_fft[out_fft_slices] = torch.einsum('bco...,cio...->bci...', 
                    x_fft_subset, 
                    weight_subset.conj() if complex_data else weight_subset)

            if len(dims_to_fft_shift) > 0:
                grad_x_fft = torch.fft.fftshift(grad_x_fft, dim=dims_to_fft_shift)

            if complex_data:
                grad_x = torch.fft.ifftn(grad_x_fft, s=x.shape[2:], dim=fft_dims, norm=fft_norm)
            else:
                grad_x = torch.fft.irfftn(grad_x_fft, s=x.shape[2:], dim=fft_dims, norm=fft_norm)

        # Gradient w.r.t. weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            if separable:
                grad_weight[weight_slices] = torch.einsum('bci...,bc...->ci...', 
                    x_fft[out_fft_slices], 
                    grad_fft[out_fft_slices].conj() if complex_data else grad_fft[out_fft_slices])
            else:
                grad_weight[weight_slices] = torch.einsum('bci...,bco...->cio...', 
                    x_fft[out_fft_slices], 
                    grad_fft[out_fft_slices].conj() if complex_data else grad_fft[out_fft_slices])
            
            # Apply mask to gradient if provided
            if mask is not None:
                grad_weight = grad_weight * mask

        # Gradient w.r.t. bias
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=tuple(range(2, grad_output.ndim)), keepdim=True)

        return grad_x, grad_weight, None, grad_bias, None, None, None, None, None

# Wrapper function
def sparse_gradient_spectral_conv(x, weight, mask, bias, fft_size, fft_dims, complex_data, fft_norm, separable):
    return SparseGradientSpectralConv.apply(x, weight, mask, bias, fft_size, fft_dims, complex_data, fft_norm, separable)