ds_config = {
  "zero_optimization": {
      "stage": 3,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "sub_group_size": 1e9,
      "reduce_bucket_size": "auto",
      "stage3_prefetch_bucket_size": "auto",
      "stage3_param_persistence_threshold": "auto",
      "stage3_max_live_parameters": 1e9,
      "stage3_max_reuse_distance": 1e9,
      "stage3_gather_16bit_weights_on_model_save": False,
      # "zero_quantized_weights": True,
      #   "zero_hpz_partition_size": 4,
      #   "zero_quantized_gradients": True,

#       "offload_optimizer": {
#           "device": "cpu",
#           "pin_memory": True
# },
#       "offload_param": {
#             "device": "cpu",
#             "pin_memory": True
# },
  },
  
  "fp16": {
      "enabled": False,
      "auto_cast": "auto",
      "loss_scale": 10,
      "initial_scale_power": 32,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
  },
  # "optimizer": {
  #   "type": "AdamW",
  #   "params": {
  #     "lr": 1e-5,
  #     "weight_decay": 0.05,
  #     "torch_adam": True,
  #     "adam_w_mode": True
  #   }
  # },
  "train_batch_size": 256,
  "gradient_accumulation_steps": 1,
  "wall_clock_breakdown": False
}


test_ds_config ={
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 1,
  "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
},
  "optimizer": {
        "type": "AdamW",
    },
  "gradient_clipping": 0.5,
}