# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    import matplotlib.pyplot as plt

    MPL_EXISTS = True
except ImportError:
    MPL_EXISTS = False


def plot_quantized_mapping(model, optimizer, tb_writer, elapsed_steps, max_n_plots=9):
    if not MPL_EXISTS:
        return

    n_plots = 0
    for n, p in model.named_parameters():
        if n_plots >= max_n_plots:
            break

        state = optimizer.state[p]
        if "latent" not in state or p.dim() < 2:
            continue

        u = state["latent"].detach()
        p = p.detach()

        # handle per-channel quantization by selecting last slice of dim 0
        u = u[-1].cpu().flatten()
        p = p[-1].cpu().flatten()

        fig = plt.figure(figsize=(5, 3))
        plt.scatter(u, p, marker=".")

        for attr in ("_orig_mod", "module"):
            n = n.rsplit(f"{attr}.", 1)[-1]
        tb_writer.add_figure("proxmap/{0}".format(n), fig, elapsed_steps)
        n_plots += 1
