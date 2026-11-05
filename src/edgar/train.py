"""Single-GPU training of the full six-expert EDGAR refiner."""

import argparse
import json
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sapiens", required=True)
    p.add_argument(
        "--poster",
        required=True,
        help="Complete affectnet8-sd.pth feature-network state",
    )
    p.add_argument("--output", required=True, help="New run directory")
    p.add_argument(
        "--base-model",
        help="Override the checkpoint's SD-Turbo location with a model ID or local directory",
    )
    p.add_argument(
        "--initialize",
        help="Complete EDGAR checkpoint to fine-tune; starts a new optimizer",
    )
    p.add_argument(
        "--resume",
        help="Resume a release training checkpoint, including optimizer and RNG",
    )
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-samples", type=int, default=16)
    p.add_argument("--lambda-expression", type=float, default=8.0)
    p.add_argument(
        "--lambda-balance",
        type=float,
        default=0.002,
        help="Value in supplied training code; paper reports 0.001. See docs/implementation.md.",
    )
    a = p.parse_args()
    if (
        min(a.steps, a.batch_size, a.accumulation, a.save_every, a.eval_samples) < 1
        or a.workers < 0
        or a.warmup < 0
        or a.eval_every < 0
    ):
        p.error("Invalid training interval, batch size, or worker count.")
    if a.initialize and a.resume:
        p.error("Choose --initialize or --resume, not both.")
    if Path(a.output).exists():
        p.error("Choose a new output directory; existing runs are never overwritten.")
    return a


def main():
    args = arguments()
    import random
    import numpy as np
    import torch
    from torch.nn import functional as F
    import torchvision
    from torchvision.transforms import functional as TF
    import lpips
    from .data import TripletDataset
    from .losses import gram_loss, load_expression_state
    from .model import EDGAR
    from .vendor.poster.PosterV2_8cls import pyramid_trans_expr2

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = TripletDataset(args.manifest, "train")
    val_data = TripletDataset(args.manifest, "validation") if args.eval_every else None
    # Resume restores optimization/RNG state; prefetched data order is not replayed.
    sampler_generator = torch.Generator().manual_seed(args.seed)
    sampler = torch.utils.data.RandomSampler(
        train_data,
        replacement=True,
        num_samples=args.steps * args.batch_size * args.accumulation,
        generator=sampler_generator,
    )
    loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = EDGAR(
        sapiens_path=args.sapiens,
        checkpoint=args.resume or args.initialize,
        base_model=args.base_model,
    ).to(dtype=dtype)
    model.set_train()
    lpips_net = lpips.LPIPS(net="vgg").cuda().eval().requires_grad_(False)
    vgg = (
        torchvision.models.vgg16(weights="IMAGENET1K_V1")
        .features.cuda()
        .eval()
        .requires_grad_(False)
    )
    poster = pyramid_trans_expr2()
    poster_state = torch.load(args.poster, map_location="cpu", weights_only=True)
    load_expression_state(poster, poster_state)
    poster = poster.cuda().eval().requires_grad_(False)
    # Include every intended trainable parameter, including the reference projection.
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, args.warmup)) if args.warmup else 1.0,
    )
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "training_state" not in state:
            raise ValueError(
                "Checkpoint has no release training state; use --initialize for a new run."
            )
        training = state["training_state"]
        optimizer.load_state_dict(training["optimizer"])
        scheduler.load_state_dict(training["scheduler"])
        start = training["step"]
        torch.set_rng_state(training["torch_rng"])
        torch.cuda.set_rng_state_all(training["cuda_rng"])
        random.setstate(training["python_rng"])
        sampler_generator.set_state(training["sampler_rng"])
    if start >= args.steps:
        raise ValueError("--steps must exceed the checkpoint's completed steps.")
    output = Path(args.output)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)

    def normalize(x):
        return TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def save(step):
        state = model.checkpoint_state()
        state["training_state"] = dict(
            step=step,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),
            sampler_rng=sampler_generator.get_state(),
        )
        destination = output / f"step_{step:06d}.pt"
        temporary = output / f"step_{step:06d}.tmp"
        torch.save(state, temporary)
        temporary.replace(destination)

    for step in range(start + 1, args.steps + 1):
        metrics = dict(l2=0.0, lpips=0.0, gram=0.0, expression=0.0, balance=0.0)
        for _ in range(args.accumulation):
            batch = next(iterator)
            source, target, reference = [
                batch[k].to("cuda", dtype=dtype)
                for k in ("input", "target", "reference")
            ]
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16
            ):
                pred = model(source, reference)
                # Preserve the supplied implementation's 384x416 crop at (64,64).
                pc, tc = pred[:, :, 64:448, 64:480], target[:, :, 64:448, 64:480]
                losses = {
                    "l2": F.mse_loss(pc.float(), tc.float())
                    + 0.1 * F.mse_loss(pred.float(), target.float()),
                    "lpips": lpips_net(pred.float(), target.float()).mean(),
                }
                top, left = random.randint(0, 32), random.randint(0, 32)
                p01, t01 = (
                    (pc.float() * 0.5 + 0.5).clamp(0, 1),
                    (tc.float() * 0.5 + 0.5).clamp(0, 1),
                )
                losses["gram"] = gram_loss(
                    normalize(p01)[:, :, top : top + 352, left : left + 384],
                    normalize(t01)[:, :, top : top + 352, left : left + 384],
                    vgg,
                )
                pf = poster(normalize(TF.resize(p01, [224, 224], antialias=True)))
                with torch.no_grad():
                    tf = poster(normalize(TF.resize(t01, [224, 224], antialias=True)))
                losses["expression"] = (
                    F.mse_loss(pf.float(), tf.float()) * args.lambda_expression
                )
                aux = pred.new_zeros(())
                for module in model.unet.modules():
                    if hasattr(module, "current_aux_loss"):
                        aux = aux + module.current_aux_loss
                        module.current_aux_loss = 0.0
                losses["balance"] = aux * args.lambda_balance
                loss = sum(losses.values()) / args.accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}.")
            loss.backward()
            for k, v in losses.items():
                metrics[k] += float(v.detach()) / args.accumulation
        projection_grad = model.ref_proj.weight.grad
        metrics["reference_projection_grad_norm"] = (
            float(projection_grad.float().norm())
            if projection_grad is not None
            else 0.0
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.update(step=step, gradient_norm=float(gradient_norm))
        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            scores, perceptual_scores = [], []
            # Keep evaluation sampling from changing subsequent training RNG state.
            with (
                torch.random.fork_rng(devices=[torch.cuda.current_device()]),
                torch.inference_mode(),
            ):
                torch.manual_seed(args.seed)
                for i in range(min(args.eval_samples, len(val_data))):
                    b = val_data[i]
                    x, r, t = [
                        b[k][None].to("cuda", dtype=dtype)
                        for k in ("input", "reference", "target")
                    ]
                    y = model(x, r)
                    scores.append(float(F.mse_loss(y.float(), t.float())))
                    perceptual_scores.append(
                        float(lpips_net(y.float(), t.float()).mean())
                    )
            model.train()
            metrics["validation_l2"] = sum(scores) / len(scores)
            metrics["validation_lpips"] = sum(perceptual_scores) / len(
                perceptual_scores
            )
        print(json.dumps(metrics), flush=True)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            save(step)


if __name__ == "__main__":
    main()
