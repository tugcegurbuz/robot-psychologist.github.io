---
layout: post
title: "Stop Letting Your GPU Nap: Stack Jobs and Supercharge Your Experiments"
categories: compute
---

*Tips for ML researchers on shared clusters who are tired of slow experiments and sleepy GPUs.*

---

### Wait, Why Is My GPU So Bored? 🥹

Ever peeked at `nvidia-smi` mid-training and felt personally offended by a **15% GPU utilization** reading?

You’re not alone.

In many ML setups—especially in deep reinforcement learning or self-supervised learning—the GPU ends up spending more time **waiting around** than doing actual work. Here's why:

- Your model might be **tiny** (looking at you, MLPs and small CNNs).
- **Environment steps** in RL live on the CPU and take their sweet time.
- **Data augmentation** and preprocessing often clog the CPU while the GPU twiddles its thumbs.
- Even classic vision or SimCLR jobs on CIFAR-10 barely dent the surface of a modern A100’s power.

Moral of the story? **You’ve got untapped compute just sitting there.**


###  Signs of GPU Underuse

Here’s how to know your GPU’s taking a nap:

- `nvidia-smi` shows **plenty of free VRAM** (e.g., using 5 GB out of 40 GB).
- Compute “Util” column idles in the teens while the CPU sits near 100 %.
  - Example: a fastai ResNet-18 computer-vision run on an A100 sat at ~20 % util with memory to spare ([reference](https://stackoverflow.com/questions/75553862/low-utilization-of-the-a100-gpu-with-fastai)) or an RLlib DQN job with 256 k batch size still spiked only briefly above 25 %

You might be tempted to buy more GPUs. Don’t. **Use what you already have better.**


### The Secret: Run Multiple Jobs at Once

If your current job is only using a slice of the GPU, just stack more on top!

Here’s the magic formula:

```bash
# Run three jobs in parallel
for cfg in cfg1.yaml cfg2.yaml cfg3.yaml; do
    python train.py --config $cfg & 
done
wait # Let them all finish before exiting
````

Why it works:

- Each job uses a slice of VRAM; their peaks rarely coincide.

- Streaming Multiprocessor stay busier because when one job waits on the CPU, another is mid-backprop.
  - **More info on SMs:** Each SM handles the actual math operations (like matrix multiplies and convolutions). A100 has 108 SMs, which means it can handle a lot of parallel math — if you feed it well.

- You triple sweep throughput without touching the cluster queue.

This trick works great for:

* Hyperparameter sweeps
* Seed averaging
* Trying three ideas because you’re impatient (relatable)


### Tips, Pitfalls, and Gotchas (With Explanations!)

| ✅ / ⚠️ | What You Should Know                                        | Why It Matters                                                                                                                                     |
| ------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✅      | **Leave \~10% VRAM unused**                                 | PyTorch loves to surprise you with memory spikes. A small buffer helps you avoid sudden OOM crashes that wipe out *all* jobs.                      |
| ✅      | **Use `/scratch` or SSD storage**                           | If three jobs all hit the disk at once on slow storage, your fancy parallelism will turn into a data-loading traffic jam.                          |
| ✅      | **Tag runs in your logger (e.g., `wandb --group stacked`)** | Keeps your dashboards from looking like a spaghetti bowl of metrics. Easier to compare, track, and brag about.                                     |
| ✅      | **Watch `num_workers` and threads**                         | Each job spawns data loaders. Multiply that by three and suddenly your system has 48 zombie processes hoarding RAM. Keep things lean.              |
| ⚠️     | **Don’t stack giant models**                                | If you’re running LLMs, ViTs, or anything eating 80%+ VRAM, just... don’t. You’ll get out-of-memory errors faster than you can say “SIGKILL”.      |
| ⚠️     | **Know your cluster’s rules**                               | Some clusters have strict policies: one job per GPU, no background processes, etc. Break them, and you might lose access. Nobody wants that email. |


### TL;DR 💛

**If your GPU looks bored, it probably is.**

Instead of leaving it idle, stack 2–3 light-to-medium jobs on the same card. You’ll:

* Finish sweeps 2–3x faster
* Reduce total GPU-hours
* Help your labmates get off the waitlist



### Your Move 💅

1. Fire up few extra jobs.
2. Monitor `nvidia-smi`.
3. Watch your GPU actually break a sweat.
4. Flex your productivity gains.

You don’t need more compute—you just need to **use it smarter**.
