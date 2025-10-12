from typing import Any, Tuple, Dict, Sequence, Optional
import mlx.core as mx
import mlx.nn as nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return mx.where(
        x < 0,
        1/(1-x + epsilon),
        x + 1
    )


def log_stablemax(x, axis=-1):
    s_x = s(x)
    return mx.log(s_x / mx.sum(s_x, axis=axis, keepdims=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.astype(mx.float64), axis=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = mx.where(valid_mask, labels, 0)
    prediction_logprobs = mx.take_along_axis(logprobs, transformed_labels.astype(mx.int32)[..., None], axis=-1)[..., 0]

    return -mx.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Manual cross entropy implementation
    logits = logits.astype(mx.float32)
    labels = labels.astype(mx.int32)
    
    # Compute log softmax manually: log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits - max_logits)
    sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = (logits - max_logits) - mx.log(sum_exp)
    
    # Get the log probability of the correct class
    batch_size, seq_len = labels.shape
    vocab_size = logits.shape[-1]
    
    # Create one-hot encoding for labels
    labels_flat = labels.reshape(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    log_probs_flat = log_probs.reshape(-1, vocab_size)
    
    # Get log probabilities for correct labels
    correct_log_probs = mx.take_along_axis(log_probs_flat, labels_flat[..., None], axis=-1)[..., 0]
    
    # Mask out ignored indices
    mask = (labels_flat != ignore_index)
    loss = -mx.where(mask, correct_log_probs, 0.0)
    
    return loss.reshape(batch_size, seq_len)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def __call__(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, mx.array, Dict[str, mx.array], Optional[Dict[str, mx.array]], mx.array]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Preds
        outputs["preds"] = mx.argmax(outputs["logits"], axis=-1)

        # Correctness
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mx.sum(mask, axis=-1)
        loss_divisor = mx.maximum(loss_counts, 1)[..., None]  # Avoid NaNs in division

        is_correct = mask & (mx.argmax(outputs["logits"], axis=-1) == labels)
        seq_is_correct = mx.sum(is_correct, axis=-1) == loss_counts
        
        # Metrics (halted)
        valid_metrics = new_carry.halted & (loss_counts > 0)
        metrics = {
            "count": mx.sum(valid_metrics),
            
            "accuracy": mx.sum(mx.where(valid_metrics, mx.sum(is_correct.astype(mx.float32) / loss_divisor, axis=-1), 0)),
            "exact_accuracy": mx.sum(valid_metrics & seq_is_correct),

            "q_halt_accuracy": mx.sum(valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)),
            "steps": mx.sum(mx.where(valid_metrics, new_carry.steps, 0)),
        }

        # Losses
        lm_loss = mx.sum(self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) * mask / loss_divisor)
        # Manual binary cross entropy: -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
        q_halt_logits = outputs["q_halt_logits"]
        targets = seq_is_correct.astype(q_halt_logits.dtype)
        sigmoid_logits = 1.0 / (1.0 + mx.exp(-q_halt_logits))  # sigmoid
        q_halt_loss = mx.sum(-(targets * mx.log(sigmoid_logits + 1e-8) + (1 - targets) * mx.log(1 - sigmoid_logits + 1e-8)))
        metrics.update({
            "lm_loss": lm_loss,
            "q_halt_loss": q_halt_loss,
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = nn.binary_cross_entropy(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss

        # Filter outputs for return
        detached_outputs = {k: outputs[k] for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, mx.all(new_carry.halted)
