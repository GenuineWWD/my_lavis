ds_config ={
  "train_batch_size": 4,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": 7000000,
    "allgather_bucket_size": 7000000,
    "reduce_scatter": True,
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": True,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
