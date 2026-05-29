from __future__ import annotations

import torch


def _sglang_bias_dropout_add(x_with_bias, residual, prob, training):
    x, bias = x_with_bias
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    return out.float() + residual.float()


def get_sglang_bias_dropout_add(training, fused):
    del fused

    def _bias_dropout_add(x_with_bias, residual, prob):
        return _sglang_bias_dropout_add(x_with_bias, residual, prob, training)

    return _bias_dropout_add
