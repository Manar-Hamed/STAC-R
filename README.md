# STAC-R

**Subspace-Tracked Activation Compression with Residuals** · NeurIPS 2026  
Manar Hamed and Yassine Yaakoubi · [Repository](https://github.com/Manar-Hamed/STAC-R)

STAC-R stores low-rank activation coefficients and sparse residuals for backward.
The linear wrappers preserve forward outputs and input gradients; weight gradients use the compressed activation.

## Install and try

Requires Python 3.10+ and PyTorch 2.4+. For GPU use, install a CUDA-enabled
PyTorch build, then install STAC-R from this directory:

```sh
python -m pip install .
python -m stacr doctor
python -m stacr train --config synthetic_stacr
```

The quick start uses Turbo-fast. For full STAC-R, use `--config synthetic_full`.

## Use in your model

```python
from stacr import apply_stacr, set_step

layers = apply_stacr(model, surface="mlp", rank=8, residual_budget=0.005)
optimizer = ...  # your usual optimizer
for step, batch in enumerate(loader):
    set_step(layers, step)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

`apply_stacr` defaults to Turbo-fast: after 50 optimizer steps, it freezes the
basis and omits residuals while model parameters keep training. Set
`turbo_after_step=None` for full STAC-R, or change the number to choose when Turbo starts.

For custom modules, pass `target_names=["encoder.projection", ...]`; for a single
`nn.Linear`, use `STACRLinear.from_linear(layer, rank=8, turbo_after_step=50)`. Wrap before DDP;
construct DDP with `broadcast_buffers=False` to keep activation bases local to each rank.
Parameters retain their identity. Save the wrapped model's `state_dict` and apply
the same configuration before loading it.
With gradient accumulation, call `set_step(layers, optimizer_step,
microbatch_index=micro, accum_steps=accum)` before each microbatch.
See [`examples/custom_model.py`](examples/custom_model.py) for a runnable example.

| Variant | Setting |
|---|---|
| Turbo-fast (default) | `turbo_after_step=50` |
| Full STAC-R with sparse correction | `turbo_after_step=None` |
| Full STAC-R with global magnitude selection | `turbo_after_step=None, residual_mode="global_topk"` |
| Sparse controls | `turbo_after_step=None`, with `residual_mode="uniform"`, `"block_sparse"`, or `"sampled"` |
| Low-rank only with a tracked basis | `turbo_after_step=None, residual_mode="none"` |
| Attention projections too | `surface="mlp+attention"` |

The adaptive implementation also supports quantized coefficients, adaptive ranks,
and residual-triggered refresh: `from stacr import apply_adaptive_stacr`.
Each experiment specifies its full configuration. Chunk selection and global
selection are different protocols; Turbo-fast keeps model parameters trainable.
Checkpoint selected blocks and compress the remaining blocks with
`stacr.composition`; full checkpointing can remove the surface STAC-R would save.
The boundary-compression protocols use reconstructed inputs during recomputation
and can also change input gradients.

## Experiments and results

Several training and memory protocols require an 8-bit optimizer. Install
the comparator dependencies before running those protocols:

```sh
python -m pip install ".[train,benchmarks,comparators]"
python -m stacr train --config gpt2_medium_turbofast --steps 500
python -m stacr reproduce list
python -m stacr reproduce run pythia-10k-full --dry-run
python -m stacr reproduce run pythia-10k-full
python -m stacr reproduce openwebtext --dry-run
python -m stacr reproduce bounds --help
python -m experiments.benchmarks list
python -m experiments.benchmarks run composed_frontier --dry-run
python -m experiments.benchmarks run composed_frontier --output outputs/frontier
python -m scripts.make_tables --output outputs/tables
python -m scripts.evidence cohorts --output outputs/recomputed-cohorts.json
python -m scripts.evidence summary --output outputs/results.csv
python -m paper_data.build.build_all --output-dir outputs/figures
```

Experiments cover quality, gradient fidelity, memory/throughput, residual and
factorization controls, model families, checkpoint/offload composition, LoRA/QLoRA,
and distributed training. Use `--help` for supported overrides. Optional optimizer
and adapter experiments need `pip install ".[galore,peft,comparators]"`; graph experiments need
`pip install ".[graph]"`. Gated models need
your own Hugging Face access. Missing dependencies raise an error.
SubTrack++ needs its complete external checkout because its pip package omits a required helper. On Linux, clone it beside this repository and select the pinned source:

```bash
git clone https://github.com/criticalml-uw/SubTrack.git ../SubTrack
git -C ../SubTrack fetch origin aabebe310e95a709454d32ce4c697d09114edac9
git -C ../SubTrack checkout aabebe310e95a709454d32ce4c697d09114edac9
export STACR_SUBTRACK_SOURCE="$(cd ../SubTrack && pwd)"
```
Launch two-GPU profiles with `torchrun --nproc_per_node=2 -m stacr reproduce run tensor-parallel-full`.

`reproduce` runs the experiment data preparation, optimizer and learning-rate schedules;
`train` is the small configurable integration runner. The adaptive engine is selectable
there with `--set engine=adaptive`; its options go in a YAML `compression` mapping.

Experiment measurements are in `paper_data`. Memory and throughput depend on
hardware, precision, batch size, and the full training configuration.

## Citation

```bibtex
@inproceedings{hamed2026stacr,
  title     = {{STAC-R}: Subspace-Tracked Activation Compression with Residuals},
  author    = {Hamed, Manar and Yaakoubi, Yassine},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
