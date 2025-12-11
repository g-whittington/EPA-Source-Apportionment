# Programmer: George Whittington
# Date: 2025-12-10
# Purpose: Extract repeated figures to functions

import polars as pl
import polars.selectors as cs

import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Add in doc strings and comment code

def extreme_values_table(df, name):
    return (
        df
        .unpivot(index="date", variable_name="species", value_name="concentration")
        .group_by("species")
        .agg(
            pl.col("concentration").mean().alias("mean"),
            pl.col("concentration").max().alias("max")
        )
        .with_columns(
            (pl.col("max") / pl.col("mean")).alias("max_mean_ratio")
        )
        .sort("max_mean_ratio", descending=True)
        .head(5)
        .select("species", "mean", "max", "max_mean_ratio")
        .style
        .fmt_number(cs.numeric(), decimals=4)
        .tab_header(
            title=f"{name} Extreme Value Audit",
            subtitle="Top 5 Species by Max-to-Mean Ratio"
        )
        .opt_row_striping()
    )

def log_comparison_plot(df_raw, df_log, target_col):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: Raw Data (Right Skewed)
    sns.histplot(data=df_raw, x=target_col, ax=axs[0], bins=30)
    axs[0].set_xlabel(f"{target_col} (Raw)")
    axs[0].set_title("Raw Data")

    # Plot 2: Log Data (Normal-ish)
    sns.histplot(data=df_log, x=target_col, ax=axs[1], bins=30)
    axs[1].set_xlabel(f"{target_col} (Log Scale)")
    axs[1].set_title("Log Transformed")

    fig.suptitle(f"Distribution of {target_col.upper()} (Raw vs Log-Transformed)")

    plt.tight_layout()

    return (fig, axs)

def summary_statistics(df, num_value=13):
    return (
        df
        .unpivot(index="date", variable_name="species", value_name="concentration")
        .group_by("species")
        .agg(
            pl.col("concentration").mean().alias("mean"),
            pl.col("concentration").median().alias("median"),
            # calculate % zeros by taking the mean of a boolean expression (True=1, False=0)
            (pl.col("concentration") == 0).mean().alias("pct_zeros")
        )
        .sort("mean", descending=True)
        .head(num_value)
        .with_columns(
            pl.col("pct_zeros")
        )
        .style
        .tab_header("Summary Statistics and Sparsity Check")
        .fmt_number(columns=["mean", "median"], decimals=4)
        .fmt_percent(columns=["pct_zeros"], decimals=1)
        .opt_row_striping()
    )

def uncertainty_ratio_plot(df, name):
    # Filter out extreme ratios (>1.0) just for plotting clarity
    plot_data = (
        df
        .filter(pl.col("uncertainty_ratio_raw") < 1.0)
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.boxplot(
        data=plot_data,
        x="uncertainty_ratio_raw",
        y="species",
        orient="h",
        fliersize=1,
        linewidth=0.8,
        ax=ax
    )

    ax.set_title(f"{name}: Sensor Reliability Audit")
    ax.set_xlabel("Uncertainty Ratio (Uncertainty / Concentration)")
    ax.axvline(x=0.2, color='r', linestyle='--', alpha=0.5, label="20% Error Threshold")
    ax.legend(loc="lower right")

    return (fig, ax)

def correlation_heatmap_plot(df, name, num_values=0, annot=False):
    full_corr = (
        df
        .select(cs.numeric())
        .to_pandas()
        .corr()
    )

    if num_values != 0:
        annot = True

        top_vars = (
            full_corr
            .stack()
            # Remove self-correlations (A vs A) and duplicates (A vs B == B vs A)
            .loc[lambda x: x.index.get_level_values(0) < x.index.get_level_values(1)]
            # Sort by absolute strength
            .sort_values(key=abs, ascending=False)
            .head(num_values)
            # Get the names from the MultiIndex (level 0 and level 1)
            .index
            .to_flat_index()
        )

        interesting_cols = list(set([item for pair in top_vars for item in pair]))

        full_corr = full_corr.loc[interesting_cols, interesting_cols]

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        full_corr,
        cmap='RdBu_r',
        center=0,
        square=True,
        cbar_kws={"shrink": .5},
        annot=annot,
        ax=ax
    )

    ax.set_title(f"{name}: Species Correlation Matrix")
    ax.invert_yaxis()

    return (fig, ax)

def pairwise_scatter_plot(df, title, target_species):
    sns.pairplot(
        df
        .select(target_species)
        .to_pandas(),
        plot_kws={'alpha': 0.5, 's': 10},
        diag_kind='kde'
    )

    plt.suptitle(title, y=1.02)

def time_series_plot(df, y, title):
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        data=df.to_pandas(),
        x="date",
        y=y,
        alpha=0.7
    )

    ax.set_title(title)
    ax.set_ylabel("Log(Concentration + 1)")
    ax.set_xlabel("Date")

    return (fig, ax)