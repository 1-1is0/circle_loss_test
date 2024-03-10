import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import matplotlib.gridspec as gridspec
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method


def plot_umap(
    model_name,
    data_title,
    embeddings_2d,
    labels,
    figsize,
    title_fontsize,
    title_padding,
    legend_fontsize,
):

    fig, ax = plt.subplots(1, figsize=figsize)
    fig.suptitle(
        f"{data_title} UMAP representation - {model_name}",
        y=title_padding,
        fontsize=title_fontsize,
    )

    df = list(zip(embeddings_2d[:, 0], embeddings_2d[:, 1], labels))
    df = pd.DataFrame(df, columns=["x", "y", "label"])

    ax = sns.scatterplot(
        data=df, x="x", y="y", hue="label", palette="dark", linewidth=0.1
    )
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.legend(loc=0, title="Classes", prop={"size": legend_fontsize})
    ax.grid(axis="both", alpha=0.3)
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    export_name = f"{model_name}_{data_title}".lower().replace(" ", "_")
    fig.savefig(f"./figs/{export_name}")
    plt.tight_layout()
    plt.show()


def plot_input_sample(
    batch_data,
    mean=[0.49139968, 0.48215827, 0.44653124],
    std=[0.24703233, 0.24348505, 0.26158768],
    to_denormalize=True,
    figsize=(3, 3),
):

    batch_image, _ = batch_data
    batch_size = batch_image.shape[0]

    random_batch_index = random.randint(0, batch_size - 1)
    random_image = batch_image[random_batch_index]

    image_transposed = random_image.detach().numpy().transpose((1, 2, 0))
    if to_denormalize:
        image_transposed = np.array(std) * image_transposed + np.array(mean)
        image_transposed = image_transposed.clip(0, 1)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_transposed)
    ax.set_axis_off()


def plot_train_val_metric(
    report_path,
    model_name,
    metric_name,
    lst_train_val_cols,
    figsize,
    linewidth_plot,
    title,
    title_fontsize,
    title_padding,
    xy_ticks_fontsize,
    xylabel_fontsize,
    legend_fontsize,
    ylim,
    to_plot_val=False,
):

    # Read csv file
    report = pd.read_csv(report_path, index_col=0)
    # Split dataframe to train and val
    train_report = report[report["mode"] == "train"]

    # Find maximum batch_number
    last_train_batch_id = train_report["batch_index"].max()

    # Keep last batch in each epoch of dataframe
    train_report_for_plot = train_report[
        train_report.batch_index == last_train_batch_id
    ]

    if to_plot_val:
        val_report = report[report["mode"] == "val"]
        last_val_batch_id = val_report["batch_index"].max()
        val_report_for_plot = val_report[val_report.batch_index == last_val_batch_id]

    # Plot train and validation metric

    train_col_name, val_col_name = lst_train_val_cols

    if title is None:
        if to_plot_val:
            title = f"Train and Validation {metric_name} of {model_name}"
        else:
            title = f"Train {metric_name} of {model_name}"

    fig, ax = plt.subplots(1, figsize=figsize)
    fig.suptitle(f"{title}", y=title_padding, fontsize=title_fontsize)

    ax.plot(
        train_report_for_plot[train_col_name].values,
        linewidth=linewidth_plot,
        label=f"Train - {model_name}",
    )

    if to_plot_val:
        ax.plot(
            val_report_for_plot[val_col_name].values,
            linewidth=linewidth_plot,
            label=f"Validation - {model_name}",
        )

    ax.set_xlabel("Epoch", fontsize=xylabel_fontsize)
    ax.set_ylabel(metric_name, fontsize=xylabel_fontsize)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(loc=0, prop={"size": legend_fontsize})
    ax.tick_params(axis="x", labelsize=xy_ticks_fontsize)
    ax.tick_params(axis="y", labelsize=xy_ticks_fontsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if ylim:
        ax.set_ylim(ylim)
    name_for_export = f"{model_name}_{metric_name}".lower().replace(" ", "_")
    fig.savefig(f"./figs/{name_for_export}")
    plt.tight_layout()
    plt.show()


def plot_images(
    dataloader,
    dataloader_title,
    model,
    model_name,
    epsilon=0.1,
    mean=[0.49139968, 0.48215827, 0.44653124],
    std=[0.24703233, 0.24348505, 0.26158768],
    to_denormalize=True,
    title_fontsize=20,
    title_padding=1.07,
    figsize=(6, 6),
):

    batch_data = next(iter(dataloader))
    images, labels = batch_data

    if model and epsilon:
        images = images.to("cuda:0")
        model.to("cuda:0")
        images = fast_gradient_method(
            model_fn=model, x=images, eps=epsilon, norm=np.inf
        )
        images = images.to("cpu")
    # Initialize an empty list to store the selected indices
    selected_indices = []

    # Loop over each integer from 0 to 9 and select 10 random indices
    for i in range(10):
        indices_i = np.random.choice(np.where(labels == i)[0], size=10, replace=False)
        selected_indices.extend(indices_i)

    selected_images = images[selected_indices]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows=10, ncols=10, wspace=0.0, hspace=0.0, top=1, bottom=0, left=0, right=1
    )
    image_index = 0
    for row_index in range(10):
        for col_index in range(10):
            current_image = selected_images[image_index]

            image_transposed = current_image.detach().numpy().transpose((1, 2, 0))
            if to_denormalize:
                image_denormalized = np.array(std) * image_transposed + np.array(mean)
                image_denormalized = image_denormalized.clip(0, 1)
            ax = plt.subplot(gs[row_index, col_index])
            ax.imshow(image_denormalized, cmap="gray")
            ax.set_axis_off()
            image_index += 1

    if model_name:
        fig.suptitle(
            f"{dataloader_title}({model_name})",
            fontsize=title_fontsize,
            y=title_padding,
        )
        name_for_export = f"{dataloader_title}_{model_name}".lower().replace(" ", "_")
    else:
        fig.suptitle(dataloader_title, fontsize=title_fontsize, y=title_padding)
        name_for_export = f"{dataloader_title}".lower().replace(" ", "_")
    fig.savefig(f"./figs/{name_for_export}")
    plt.show()
