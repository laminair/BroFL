import matplotlib.pyplot as plt
import pandas as pd
import os

from pathlib import Path
import netrc
import wandb

import matplotlib.ticker as ticker


def download_data_from_wandb(entity: str, project: str, keywords: list):
    api_key = netrc.netrc().authenticators(host="api.wandb.ai")[2]
    api = wandb.Api(api_key=api_key, timeout=1800)
    runs = api.runs(path=f"{entity}/{project}")
    df = pd.DataFrame()

    if type(keywords) == str:
        keywords = [f"{keywords}", ]

    for run in runs:
        # We are only interested in 2 sorts of logs: server logs, and baseline logs
        for keyword in keywords:

            if type(keyword) != str:
                continue

            if keyword in run.name:
                run_scan = api.run(f"{entity}/{project}/{run.id}")
                rows = run_scan.scan_history()
                rows_pd = pd.DataFrame(rows)
                rows_pd["name"] = run.name
                rows_pd["id"] = run.id
                df = pd.concat([df, rows_pd])

    return df


def download_summary_metrics(run_ids:list, entity, project):
    api_key = netrc.netrc().authenticators(host="api.wandb.ai")[2]
    api = wandb.Api(api_key=api_key, timeout=1800)
    run_df = pd.DataFrame()

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        run_df = pd.concat([run_df, pd.DataFrame({"id": run_id, **run.summary._json_dict})])

    return run_df


def download_system_metrics(run_ids: list, entity, project):
    api_key = netrc.netrc().authenticators(host="api.wandb.ai")[2]
    api = wandb.Api(api_key=api_key, timeout=1800)
    run_df = pd.DataFrame()

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        run_df = pd.concat([run_df, pd.DataFrame({"id": run_id, "name": run.name, **run.history(stream="events")})])

    return run_df


def write_df_to_disk(df: pd.DataFrame, filename):
    data_dir = Path(f"{os.path.dirname(os.path.realpath(__file__))}")
    data_dir = Path(f"{data_dir}/data")
    data_dir.mkdir(exist_ok=True, parents=True)

    df.to_csv(path_or_buf=f"{data_dir}/{filename}")
    print(f"File {filename} written to disk.")


def expand_experiment_name(df: pd.DataFrame):
    # 2023-05-23_15:27_flbench_experiment-plan-baseline_shakespeare_lstm_None_local_1_rounds_1_clients_0_dropout_nodp_0_172.24.32.54_client_1
    col_names = ["date", "time", "experiment", "inventory", "dataset", "model", "strategy", "data_dist",
                 "training_rounds", "pl1", "clients", "pl2", "dropout", "pl3", "dp", "noise_multiplier", "ip_addr",
                 "pl4", "client_id"]

    try:
        df[col_names] = df["name"].str.split("_", expand=True)
        df.drop(["pl1", "pl2", "pl3", "pl4"], inplace=True, axis=1)
    except ValueError:
        try:
            col_names = ["date", "time", "experiment", "inventory", "dataset", "model", "strategy", "data_dist",
                         "training_rounds", "pl1", "clients", "pl2", "dropout", "pl3", "dp", "noise_multiplier", "ip_addr",
                         "client_id"]
            df[col_names] = df["name"].str.split("_", expand=True)
            df.drop(["pl1", "pl2", "pl3"], inplace=True, axis=1)
        except ValueError:
            # 2023-05-31_17:38_flbench_experiment-plan-baseline_shakespeare_lstm_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16_172.24.33.72_client_1
            print("here")
            col_names = ["date", "time", "experiment", "inventory", "dataset", "model", "strategy", "data_dist",
                         "training_rounds", "pl1", "clients", "pl2", "dropout", "pl3", "dp", "noise_multiplier", "pl4",
                         "precision", "ip_addr", "pl5", "client_id"]
            df[col_names] = df["name"].str.split("_", expand=True)
            df.drop(["pl1", "pl2", "pl3", "pl4", "pl5"], inplace=True, axis=1)

    df["timestamp"] = df[["date", "time"]].apply(" ".join, axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
    return df


def format_xaxis(ax1, ylim=None, ymax=8, with_devices=True):
    # Device type axis
    if with_devices:
        factor = 1
        ax1.set_xticklabels(
            ["GPU", "VM", "Nano", "RPi", "GPU", "VM", "Nano", "RPi", "GPU", "VM", "Nano", "RPi", "GPU", "VM", "Nano", "RPi",
             "GPU", "VM", "Nano", "RPi", "GPU", "VM", "Nano", "RPi"])
    else:
        factor = 0.5
        ax1.set_xticklabels(
            ["CNN (14k)", "LSTM (40k)", "ResNet (100k)", "DenseNet (252k)", "CNN (33k)", "LSTM (819k)"])

    # Model axis
    ax2 = ax1.twiny()
    ax2.spines["bottom"].set_position(("axes", -0.26))
    ax2.tick_params('both', length=0, width=0, which='minor', labelsize=10)
    ax2.tick_params('both', direction='in', which='major', labelsize=10)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    ax2.set_xticks([0.166, 0.333, 0.50, 0.666, 0.833])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.08, 0.246, 0.416, 0.586, 0.753, 0.913]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(
        ["CNN (14k)", "LSTM (40k)", "ResNet (100k)", "DenseNet (252k)", "CNN (33k)", "LSTM (819k)"]))

    # Dataset axis
    ax3 = ax1.twiny()

    ax3.spines["bottom"].set_position(("axes", -0.36))
    ax3.tick_params('both', length=0, width=0, which='minor', labelsize=10)
    ax3.tick_params('both', direction='in', which='major', labelsize=10)
    ax3.xaxis.set_ticks_position("bottom")
    ax3.xaxis.set_label_position("bottom")

    ax3.set_xticks([0.666, 0.833])
    ax3.xaxis.set_major_formatter(ticker.NullFormatter())
    ax3.xaxis.set_minor_locator(ticker.FixedLocator([0.3333, 0.757, 0.913]))
    ax3.xaxis.set_minor_formatter(ticker.FixedFormatter(["BLOND", "FEMNIST", "Shakespeare"]))

    # Separation line
    line_width = 2
    line_style = "dashed"
    ax1.vlines(x=3.483, ymin=0, ymax=ymax, linestyles=line_style, linewidth=line_width)
    ax1.vlines(x=3.483 * 2.16, ymin=0, ymax=ymax, linestyles=line_style, linewidth=line_width)
    ax1.vlines(x=3.483 * 3.31, ymin=0, ymax=ymax, linestyles=line_style, linewidth=line_width)
    ax1.vlines(x=3.483 * 4.45, ymin=0, ymax=ymax, linestyles=line_style, linewidth=line_width)
    ax1.vlines(x=3.483 * 5.6, ymin=0, ymax=ymax, linestyles=line_style, linewidth=line_width)
    plt.ylim(0, ymax)


def write_figure_to_disk(plt, file_name, chapter_name=None, file_type="pdf", dpi=300):
    evaluations_dir = Path(f"{os.path.dirname(os.path.realpath(__file__))}")

    if chapter_name is None:
        chapter_name = "misc"

    save_path = Path(f"{evaluations_dir}/figures/{chapter_name}")
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = Path(f"{save_path}/{file_name}.{file_type}")

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Plot saved.")
