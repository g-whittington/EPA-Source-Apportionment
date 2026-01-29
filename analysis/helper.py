# Programmer: George Whittington
# Date: 2025-12-10
# Purpose: Extract repeated figures to functions

import polars as pl
import polars.selectors as cs

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from plotnine import (
    ggplot, aes, labs, theme, 
    geom_errorbarh, geom_point,
    facet_wrap, element_text,
    theme_minimal,scale_y_discrete,
    element_blank
)

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
    ax.legend()

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

def dominant_species_stats(df_raw, city_name, target_species):
    return (
        df_raw
        .select(target_species)
        .unpivot(variable_name="species", value_name="concentration")
        .group_by("species")
        .agg(
            pl.col("concentration").mean().alias("Arithmetic Mean (µg/m³ or ppbC)"),
            pl.col("concentration").log1p().mean().alias("Log Mean (Model Unit)"),
            pl.col("concentration").max().alias("Max Value")
        )
        .with_columns(pl.lit(city_name).alias("City"))
        .select("City", "species", "Arithmetic Mean (µg/m³ or ppbC)", "Log Mean (Model Unit)", "Max Value")
    )

def plot_top5_dominant(df_list, city_names):
    top_data = []

    for df, city in zip(df_list, city_names):
        stats = (
            df
            .unpivot(index="date", variable_name="species", value_name="concentration")
            .with_columns(log_val=pl.col("concentration").log1p())
            .group_by("species")
            .agg(
                mean=pl.col("log_val").mean(),
                std=pl.col("log_val").std()
            )
            .sort("mean", descending=True)
            .head(5) 
            .with_columns(pl.lit(city).alias("City"))
        )
        top_data.append(stats)

    plot_df = pl.concat(top_data).to_pandas()

    # --- 1. Create a Unique ID for correct sorting per panel ---
    # We combine City + Species so "Sulfate" in Baltimore is distinct from "Sulfate" in St. Louis
    plot_df['unique_id'] = plot_df['species'] + "_" + plot_df['City']

    # --- 2. Sort the data so the dots appear High -> Low ---
    plot_df = plot_df.sort_values(['City', 'mean'], ascending=[True, True])
    
    # Lock this order in as a Categorical type
    plot_df['unique_id'] = pd.Categorical(
        plot_df['unique_id'], 
        categories=plot_df['unique_id'], 
        ordered=True
    )

    # --- 3. Create a Label Mapper ---
    # This dictionary tells the plot: "When you see 'Sulfate_Baltimore', just print 'Sulfate'"
    label_map = dict(zip(plot_df['unique_id'], plot_df['species']))

    # --- 4. Plot ---
    g = (
        ggplot(plot_df, aes(x="mean", y="unique_id"))
        
        # The Whiskers (Standard Deviation)
        + geom_errorbarh(aes(xmin="mean - std", xmax="mean + std"), height=0.3, color="#555555")
        
        # The Point (Mean)
        + geom_point(size=3.5, color="#2c7bb6", fill="white", stroke=1.5)
        
        # The Facets (scales="free_y" is crucial here)
        + facet_wrap("~City", scales="free_y", ncol=3)
        
        # Apply our custom label map so we don't see the ugly unique IDs
        + scale_y_discrete(labels=label_map)
        
        + labs(
            x="Log Mean Concentration (+ Std Dev)", 
            y=None, # Remove Y label since species names are self-explanatory
            title="Dominant Pollution Profile by City (Top 5 Species)"
        )
        + theme_minimal()
        + theme(
            figure_size=(12, 5),
            panel_spacing=0.4,       # Adds breathing room between the 3 plots
            strip_text=element_text(size=12, weight="bold"), # Makes City titles bigger
            axis_text_y=element_text(size=10, color="black"),
            panel_grid_major_y=element_blank() # Removes horizontal grid lines for a cleaner look
        )
    )

    return g

def plot_episodic_ratios(df_balt, df_stl, df_br):
    
    def calc_ratio(df, city, species, label):
        stats = df.select(pl.col(species)).to_pandas()
        mean_val = stats[species].mean()
        max_val = stats[species].max()
        ratio = max_val / mean_val
        return {"City": city, "Species": label, "Ratio": ratio}

    data = [
        # Baltimore
        calc_ratio(df_balt, "Baltimore", "pm2.5", "PM2.5"),
        calc_ratio(df_balt, "Baltimore", "sulfate", "Sulfate"),
        calc_ratio(df_balt, "Baltimore", "organic_carbon", "Organic Carbon"),
        
        # St. Louis
        calc_ratio(df_stl, "St. Louis", "pm2.5", "Mass (PM2.5)"),
        calc_ratio(df_stl, "St. Louis", "organic_carbon", "Organic Carbon"),
        calc_ratio(df_stl, "St. Louis", "elemental_carbon", "Elemental Carbon"),
        
        # Baton Rouge
        calc_ratio(df_br, "Baton Rouge", "n-hexane", "N-Hexane"),
        calc_ratio(df_br, "Baton Rouge", "isopentane", "Isopentane"),
        calc_ratio(df_br, "Baton Rouge", "unidentified", "Unidentified"),
    ]
    
    plot_df = pl.DataFrame(data).to_pandas()

    # 2. Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="Ratio",
        y="Species",
        hue="City",
        dodge=False,
        palette="viridis"
    )

    for _, container in enumerate(plt.gca().containers):
        plt.gca().bar_label(container, fmt='%.1fx', padding=3)

    plt.title("Episodic Pollution: Max-to-Mean Ratios across Cities")
    plt.xlabel("Variability Factor (Max / Mean)")
    plt.xlim(0, 18) # Give room for the 14.9x label
    plt.grid(True, axis='x', alpha=0.3)

def plot_baton_rouge_variability(df_log):
    target_vocs = ["tnmoc", "unidentified", "propane", "ethane", "isopentane"]
    
    # 2. Prepare data for plotting
    plot_data = (
        df_log
        .select(target_vocs)
        .unpivot(variable_name="Species", value_name="Log Concentration")
        .to_pandas()
    )


    plt.figure(figsize=(10, 6))
    
    sns.violinplot(
        data=plot_data,
        x="Log Concentration",
        y="Species",
        order=target_vocs, 
        palette="rocket",   
        linewidth=1
    )

    # Add the unit label clearly to address the "ppbC" distinction
    plt.xlabel("Log(Concentration + 1) [ppbC - Gas Phase]")
    plt.title("Baton Rouge: High Variability in Dominant VOCs")
    plt.grid(True, axis='x', alpha=0.3)