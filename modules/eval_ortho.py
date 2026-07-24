import torch
import torch.nn.functional as F
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

@torch.no_grad()
def cosine_u1_u2(u1, u2):
    u1 = u1.float(); u2 = u2.float()
    c = F.cosine_similarity(u1, u2, dim=-1)   # [N], per-sample cos
    return c.abs().mean().item()

@torch.no_grad()
def grassmann_distance(X, Y, k=10):
    X = X.float(); Y = Y.float()
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    Vx = torch.linalg.svd(Xc, full_matrices=False).Vh   
    Vy = torch.linalg.svd(Yc, full_matrices=False).Vh
    k = min(k, Vx.shape[0], Vy.shape[0])
    Qx, Qy = Vx[:k], Vy[:k]                              
    s = torch.linalg.svd(Qx @ Qy.T, full_matrices=False).S.clamp(-1, 1)  
    return torch.sin(torch.arccos(s)).mean().item()

def contour_train_test(model, data_module, name, mod1_name="Vision", mod2_name="Text"):
    """
    mod1_name / mod2_name: labels for the two input modalities, e.g. "Vision"/"Text"
    for MOSI/MOSEI/UR-FUNNY/MUStARD, or "Tabular"/"Timeseries" for MIMIC.
    """

    roles = ["Fused", mod1_name, mod2_name]
    role_key = {"Fused": "fused", mod1_name: "mod1", mod2_name: "mod2"}

    base_colors = {
        "Train-Fused": (240/255, 200/255, 40/255),
        "Test-Fused": (194/255, 153/255, 5/255),
        "Train-mod1": (255/255, 143/255, 175/255),
        "Test-mod1": (188/255, 11/255, 73/255),
        "Train-mod2": (130/255, 200/255, 240/255),
        "Test-mod2": (0/255, 152/255, 199/255),
    }
    # remap the generic mod1/mod2 keys onto the actual provided names
    color_map = {
        "Train-Fused": base_colors["Train-Fused"],
        f"Train-{mod1_name}": base_colors["Train-mod1"],
        f"Train-{mod2_name}": base_colors["Train-mod2"],
        "Test-Fused": base_colors["Test-Fused"],
        f"Test-{mod1_name}": base_colors["Test-mod1"],
        f"Test-{mod2_name}": base_colors["Test-mod2"],
    }
    style_map = {
        "Train-Fused": '-.', f"Train-{mod1_name}": '-', f"Train-{mod2_name}": '--',
        "Test-Fused": '-.', f"Test-{mod1_name}": '-', f"Test-{mod2_name}": '--',
    }

    cmap_fused = LinearSegmentedColormap.from_list("custom", [(0, (255/255, 255/255, 255/255)), (1, (255/255, 244/255, 0/255))])
    cmap_mod1 = LinearSegmentedColormap.from_list("custom", [(0, (255/255, 255/255, 255/255)), (1, (214/255, 35/255, 35/255))])
    cmap_mod2 = LinearSegmentedColormap.from_list("custom", [(0, (255/255, 255/255, 255/255)), (1, (15/255, 220/255, 190/255))])
    cmap_by_role = {"fused": cmap_fused, "mod1": cmap_mod1, "mod2": cmap_mod2}

    model.eval()
    y, tr_fused, tr_mod1, tr_mod2 = model.extract_all_the_features(data_module.train_dataloader())
    y, te_fused, te_mod1, te_mod2 = model.extract_all_the_features(data_module.test_dataloader())

    X = np.vstack([tr_fused, tr_mod1, tr_mod2, te_fused, te_mod1, te_mod2])

    modalities = (
        ['Fused'] * len(tr_fused) +
        [mod1_name] * len(tr_mod1) +
        [mod2_name] * len(tr_mod2) +
        ['Fused'] * len(te_fused) +
        [mod1_name] * len(te_mod1) +
        [mod2_name] * len(te_mod2)
    )
    splits = (
        ['Train'] * (len(tr_fused) + len(tr_mod1) + len(tr_mod2)) +
        ['Test'] * (len(te_fused) + len(te_mod1) + len(te_mod2))
    )
    combined_labels = [f"{splits[i]}-{modalities[i]}" for i in range(len(splits))]
    unique_labels = list(dict.fromkeys(combined_labels))  

    def role_of(label):
        split, mod = label.split("-", 1)
        return role_key[mod]

    umap = UMAP(n_components=2, n_neighbors=30, random_state=42, metric='cosine')
    X_umap = umap.fit_transform(X)

    class_densities = {}
    bandwidths_x, bandwidths_y = [], []
    for label in unique_labels:
        idx = [i for i, x in enumerate(combined_labels) if x == label]
        x = X_umap[idx, 0]
        y_ = X_umap[idx, 1]
        xy = np.vstack([x, y_])
        density_fn = gaussian_kde(xy)
        class_densities[label] = density_fn
        bandwidths_x.append(np.sqrt(density_fn.covariance[0, 0]))
        bandwidths_y.append(np.sqrt(density_fn.covariance[1, 1]))

    max_bw_x = max(bandwidths_x)
    max_bw_y = max(bandwidths_y)

        x_min, x_max = X_umap[:, 0].min(), X_umap[:, 0].max()
    y_min, y_max = X_umap[:, 1].min(), X_umap[:, 1].max()

    padding_frac = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= padding_frac * x_range + 3 * max_bw_x
    x_max += padding_frac * x_range + 3 * max_bw_x
    y_min -= padding_frac * y_range + 3 * max_bw_y
    y_max += padding_frac * y_range + 3 * max_bw_y

    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_min, x_max = x_center - max_span / 2, x_center + max_span / 2
    y_min, y_max = y_center - max_span / 2, y_center + max_span / 2

    xx, yy = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]
    density_map = np.zeros((300, 300, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        density_fn = class_densities[label]
        points = np.vstack([xx.ravel(), yy.ravel()])
        density_values = density_fn(points).reshape(xx.shape)
        density_map[:, :, i] = density_values

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.09, 0.08, 0.72, 0.85])
    plt.sca(ax)

    all_density_by_role = {"fused": np.zeros((300, 300)), "mod1": np.zeros((300, 300)), "mod2": np.zeros((300, 300))}
    for i, label in enumerate(unique_labels):
        all_density_by_role[role_of(label)] += density_map[:, :, i]

    global_vmax = max(m.max() for m in all_density_by_role.values())

    for i, label in enumerate(unique_labels):
        role = role_of(label)
        split = label.split("-", 1)[0]
        own_max = density_map[:, :, i].max()
        mask = density_map[:, :, i] > (0.02 * own_max)
        levels = np.linspace(0.02 * own_max, own_max, 6)
        cmap = cmap_by_role[role]

        if split == "Train":
            plt.imshow(np.where(mask.T, all_density_by_role[role].T, np.nan), cmap=cmap,
                       vmin=0, vmax=global_vmax, extent=(x_min, x_max, y_min, y_max), origin='lower')

        plt.contour(density_map[:, :, i].T, colors=[color_map[label]],
                    extent=(x_min, x_max, y_min, y_max), alpha=1, levels=levels,
                    linestyles=style_map[label], label=label)

    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")

    ax.tick_params(axis='both', which='both', labelsize=10, colors='#333333', direction='out', length=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#444444')
        spine.set_linewidth(1.0)
    ax.set_axisbelow(True)
    ax.grid(True, which='major', color='#D9D9D9', linestyle='-', linewidth=0.8)

    legend_elements = [
        Line2D([0], [0], color=color_map[label], linestyle=style_map[label], linewidth=2, label=label)
        for label in color_map
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
                     frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_alpha(1.0)

    pos = ax.get_position()
    cbar_h = pos.height * 0.30
    gap = pos.height * 0.05

    cbar_ax1 = fig.add_axes([pos.x1 + 0.02, pos.y1 - cbar_h, 0.02, cbar_h])
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_fused, norm=plt.Normalize(0, global_vmax)),
                 cax=cbar_ax1, label="Fused")

    cbar_ax2 = fig.add_axes([pos.x1 + 0.02, pos.y1 - 2*cbar_h - gap, 0.02, cbar_h])
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_mod1, norm=plt.Normalize(0, global_vmax)),
                 cax=cbar_ax2, label=mod1_name)

    cbar_ax3 = fig.add_axes([pos.x1 + 0.02, pos.y1 - 3*cbar_h - 2*gap, 0.02, cbar_h])
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_mod2, norm=plt.Normalize(0, global_vmax)),
                 cax=cbar_ax3, label=mod2_name)

    plt.savefig(name)
