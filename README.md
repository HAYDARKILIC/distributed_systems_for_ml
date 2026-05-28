# distributed_systems_for_ml

A six-week, first-principles course on **Distributed Systems for Machine Learning**. Each week is a self-contained Jupyter Notebook that derives the underlying systems concepts from scratch, implements them in pure Python/NumPy (with small simulators in place of real multi-node clusters), visualizes the behavior, and reinforces the theory with exercises.

The goal is not to teach a single framework's API, but to expose the *mechanics* underneath PyTorch DDP, FSDP, DeepSpeed, Megatron-LM, and parameter-server systems — so that you understand *why* they make the trade-offs they do.

## Philosophy

- **From first principles.** Every collective communication primitive, sharding scheme, and synchronization protocol is rebuilt in NumPy before any real library is mentioned.
- **Simulated, not hand-wavy.** Multi-worker behavior is reproduced with `multiprocessing` / `threading` and event-driven simulators so the dynamics are concrete and reproducible on a single laptop.
- **Visualization-first.** Communication topologies, scaling curves, gradient-staleness effects, and memory-vs-throughput trade-offs are all plotted.
- **Self-contained.** Each notebook runs top-to-bottom with only NumPy, Matplotlib, NetworkX, and the standard library.

## Curriculum

| Week | Notebook | Topic |
|------|----------|-------|
| 1 | `week1_foundations_and_communication.ipynb` | Distributed systems foundations: processes, messages, latency/bandwidth cost models, the α–β model, and a from-scratch message-passing simulator |
| 2 | `week2_collective_communication.ipynb` | Collective primitives from scratch: broadcast, reduce, all-reduce (naive vs ring vs recursive-halving/doubling), all-gather, reduce-scatter; bandwidth-optimality analysis |
| 3 | `week3_data_parallelism.ipynb` | Synchronous SGD, gradient all-reduce, gradient accumulation, large-batch training (linear scaling rule, LARS/LAMB intuition), weak vs strong scaling |
| 4 | `week4_async_and_parameter_servers.ipynb` | Parameter-server architecture, asynchronous SGD, gradient staleness, Hogwild!, consistency models, and bounded-staleness simulators |
| 5 | `week5_model_and_pipeline_parallelism.ipynb` | Tensor (intra-layer) parallelism, pipeline parallelism (GPipe vs 1F1B), the bubble overhead, ZeRO/FSDP sharding, and a memory accountant |
| 6 | `week6_fault_tolerance_and_scaling.ipynb` | Checkpointing, failure models, elastic training, gradient compression (quantization, top-k sparsification, error feedback), and a capstone: a complete simulated distributed training run |

## Requirements

```bash
pip install numpy matplotlib networkx
```

Python 3.10+. No GPU required — everything is simulated.

## How to use

Work through the notebooks in order; each builds on the communication primitives and abstractions defined in the previous one. Exercises at the end of each notebook range from short derivations to small implementation challenges.
