# Implementation details

## Network

The full EDGAR model uses six experts in each of 11 independent MoE modules.
The validated SD-Turbo base snapshot is
`b261bac6fd2cf515557d5d0707481eafa0485ec2`. Preserve this snapshot for comparisons.
Using zero-based UNet indices, there are two modules in each of down_blocks[0]
and down_blocks[1], one in mid_block, and three in each of up_blocks[2] and
up_blocks[3]. Each module selects one expert per sample. The transformer FFNs in
down_blocks[2] and up_blocks[1] remain ordinary FFNs.

The released loader requires the full UNet state, VAE LoRA/skip weights, and
reference projection. It rejects incomplete or incompatible states. The exported
`edgar_exp6.pt` preserves the inference tensors of the recovered 40,001-step
six-expert application checkpoint; optimizer state is excluded. The inspected
CAP4D and RGBAvatar launchers use this checkpoint. The separate historical v4.1
checkpoint has the same architecture but different weights.

## Preprocessing

| Profile | Render input | Reference input |
| --- | --- | --- |
| training | Resize to 512², range [-1,1] | Resize to 1024²; `(RGB × 255 − mean) / std` |
| application | Symmetric white square padding, 512² bilinear resize, range [-1,1] | PIL resize to 1024², range [-1,1] |

The Sapiens mean is `[123.5, 116.5, 103.5]` and std is `[58.5, 57.0, 57.5]`.
These two reference conventions occur in the recovered training and application
sources respectively. They are explicit options to preserve the historical behavior.
Newly trained weights should use the training convention. Image inference defaults
to `training`; native avatar refinement defaults to `application`.

The model samples its VAE latent. Use the same seed, precision, input order, and
software environment for repeatable comparisons. FP32 is the inference default.

## Training settings and source corrections

The recovered implementation uses timestep 249, prompt `High resolution portrait`,
and load-balancing weight 0.002. The manuscript reports timestep 250 and balancing
weight 0.001. The code retains the recovered runtime values; these are not silently
changed during cleanup. `--lambda-balance` controls the balancing coefficient.

The pixel objective retains the source's 384×416 crop at (64,64), plus 0.1 times
full-frame MSE. The retained Gram implementation couples batch and channel
dimensions. The expression loss uses frozen POSTER feature tokens.

The cleaned trainer includes the reference projection in the optimizer, fixes a
validation channel-selection error, keeps feature extractors in evaluation mode,
and saves complete checkpoints. These corrections mean new training runs are not
claimed to replay the original training trajectory exactly. Resume restores model,
optimizer, scheduler, and Torch/Python RNG states; data-prefetch order is not replayed.

Training requires an explicit manifest. The recovered file-order split and a new
identity-disjoint split are both distributed and named in the dataset documentation.
