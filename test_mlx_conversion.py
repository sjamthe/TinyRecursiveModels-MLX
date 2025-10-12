#!/usr/bin/env python3
"""
Test script to verify the MLX conversion works correctly.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from models.recursive_reasoning.trm_mlx import TinyRecursiveReasoningModel_ACTV1
from models.losses_mlx import ACTLossHead, softmax_cross_entropy


def test_model_creation():
    """Test that we can create the model successfully."""
    print("Testing model creation...")
    
    config = {
        'batch_size': 2,
        'seq_len': 81,
        'puzzle_emb_ndim': 64,
        'num_puzzle_identifiers': 10,
        'vocab_size': 10,
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 1,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 5,
        'halt_exploration_prob': 0.1,
        'forward_dtype': 'float32',
        'mlp_t': False,
        'puzzle_emb_len': 16,
        'no_ACT_continue': True
    }
    
    try:
        model = TinyRecursiveReasoningModel_ACTV1(config)
        model.eval()  # Set to evaluation mode to avoid sparse embedding training issues
        print("✅ Model creation successful")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_forward_pass(model):
    """Test a forward pass through the model."""
    print("Testing forward pass...")
    
    if model is None:
        print("❌ Cannot test forward pass - model is None")
        return False
    
    try:
        # Create dummy batch
        batch_size = 2
        seq_len = 81
        batch = {
            'inputs': mx.random.randint(0, 10, (batch_size, seq_len)),
            'labels': mx.random.randint(0, 10, (batch_size, seq_len)),
            'puzzle_identifiers': mx.random.randint(0, 10, (batch_size,))
        }
        
        # Initialize carry
        carry = model.initial_carry(batch)
        print(f"✅ Initial carry created: steps={carry.steps.shape}, halted={carry.halted.shape}")
        
        # Forward pass
        new_carry, outputs = model(carry=carry, batch=batch)
        print(f"✅ Forward pass successful: logits={outputs['logits'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_head(model):
    """Test the loss head."""
    print("Testing loss head...")
    
    if model is None:
        print("❌ Cannot test loss head - model is None")
        return False
    
    try:
        # Wrap model with loss head
        loss_head = ACTLossHead(model, 'softmax_cross_entropy')
        
        # Create dummy batch
        batch_size = 2
        seq_len = 81
        batch = {
            'inputs': mx.random.randint(0, 10, (batch_size, seq_len)),
            'labels': mx.random.randint(0, 10, (batch_size, seq_len)),
            'puzzle_identifiers': mx.random.randint(0, 10, (batch_size,))
        }
        
        # Initialize carry
        carry = loss_head.initial_carry(batch)
        
        # Forward pass with loss
        new_carry, loss, metrics, outputs, all_finish = loss_head(
            carry=carry, 
            batch=batch, 
            return_keys=['logits']
        )
        
        print(f"✅ Loss head successful: loss={loss}, metrics={len(metrics)} items")
        return True
    except Exception as e:
        print(f"❌ Loss head failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model):
    """Test a training step with gradients."""
    print("Testing training step...")
    
    if model is None:
        print("❌ Cannot test training step - model is None")
        return False
    
    try:
        # Wrap model with loss head
        loss_head = ACTLossHead(model, 'softmax_cross_entropy')
        
        # Create dummy batch
        batch_size = 2
        seq_len = 81
        batch = {
            'inputs': mx.random.randint(0, 10, (batch_size, seq_len)),
            'labels': mx.random.randint(0, 10, (batch_size, seq_len)),
            'puzzle_identifiers': mx.random.randint(0, 10, (batch_size,))
        }
        
        # Initialize carry
        carry = loss_head.initial_carry(batch)
        
        # Define loss function
        def loss_fn(model, carry, batch):
            new_carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
            return loss, (new_carry, metrics)
        
        # Compute loss and gradients
        (loss, (new_carry, metrics)), grads = mx.value_and_grad(loss_fn)(loss_head, carry, batch)
        
        print(f"✅ Training step successful: loss={loss}, gradients computed")
        return True
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting MLX conversion tests...\n")
    
    # Test model creation
    model = test_model_creation()
    print()
    
    # Test forward pass
    forward_success = test_forward_pass(model)
    print()
    
    # Test loss head
    loss_success = test_loss_head(model)
    print()
    
    # Test training step
    training_success = test_training_step(model)
    print()
    
    # Summary
    print("📊 Test Summary:")
    print(f"  Model Creation: {'✅' if model is not None else '❌'}")
    print(f"  Forward Pass: {'✅' if forward_success else '❌'}")
    print(f"  Loss Head: {'✅' if loss_success else '❌'}")
    print(f"  Training Step: {'✅' if training_success else '❌'}")
    
    if all([model is not None, forward_success, loss_success, training_success]):
        print("\n🎉 All tests passed! MLX conversion appears to be working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
