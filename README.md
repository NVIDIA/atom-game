# Atom game: reconfiguring neutral atoms with reinforcement learning

### Summary

The goal of this project is to show how to perform atom reconfiguration (AR) with reinforcement learning on a neutral atom quantum computer. AR aims to reduce the implementation depth of a circuit and improve its fidelity.


### Prerequisites

See `requirements.txt` for the required packages.

```bash
pip install -r requirements.txt
```

That is everything the experiments in this repository need. Circuit drawing
through Qiskit is reached only by passing `tool="qiskit"` or `drawable=True` to
the generators in `qds/circuit/`, which no configuration here does; that path
imports `qwrapper`, which is not a dependency of anything else and is not
installed by the line above.

### Layout

```
qds/        the package: circuit generation, the game and its cost model, the PPO task
conf/       Hydra configurations, one per benchmark plus the transfer pools
bin/        entry points that fit or measure a policy
report/     tables and figures, read off the logs the entry points leave
scan/       cost of one decision as the instance grows
experiments/shell drivers that string the above together
tools/      release and housekeeping scripts, nothing an experiment needs
```

Anything under `bin/`, `report/` and `scan/` imports `qds`, so the repository
root has to be on the import path. The scripts in `experiments/` arrange that
themselves and may be run from any directory; invoke the others with
`PYTHONPATH=.` from the root, as the examples below do.

### Training

Both fixed-circuit and transferable PPO training use `bin/train.py`.
Launch it with `torchrun` because the trainer uses distributed PyTorch:

```bash
# Fixed circuit (the default is qnn-100)
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 bin/train.py

# Transferable policy for 50 atoms
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_50

# Transferable policy for 100 atoms
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100
```

Hydra overrides can select a fixed benchmark, for example
`benchmark.id=qft-100 board.row=20 board.col=10`.
Use `benchmark.id=qnn-lite-100` for the deterministic synthetic QNN
benchmark. It creates 343 random-matching layers with 12 two-qubit gates
per layer directly, without loading the PennyLane MQT dataset.
See `experiments/benchmark.sh` and `experiments/benchmark_transfer.sh` for complete experiment commands.

### Contributing

This project is currently not accepting contributions.

### License

Apache License 2.0; see [LICENSE](LICENSE).

The cost model in `qds/qcgame/cost.py` implements the estimator specified in
Section "Reconfiguration cost estimation" of
[arXiv:2506.05536](https://arxiv.org/abs/2506.05536). A separate reference
implementation of that estimator by Jonathan Wurtz of QuEra Computing Inc. is
published as
[QuEraComputing/reconfiguration_cost_estimator](https://github.com/QuEraComputing/reconfiguration_cost_estimator);
this repository neither includes nor requires it.

This project installs its dependencies from the Python Package Index and bundles
none of them. Their licences are reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
