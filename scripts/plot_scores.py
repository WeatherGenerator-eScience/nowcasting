import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

from nowcasting.metrics.validation import Evaluator

FIG_DIR = "./results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

def load_model_scores(
    model_names: List[str],
    dir: Optional[str] = None,
    leadtimes: List[int] = [5, 10, 15, 30, 60, 90],
    thresholds: List[float] = [0.5, 1, 2, 5, 10, 20, 30]
):
    """
    Loads pre-computed evaluation scores for multiple nowcasting models.

    This function iterates through a list of model names, initializes an `Evaluator`
    for each, loads their accumulated scores, and then retrieves categorical,
    continuous, and Fractions Skill Score (FSS) metrics.

    Parameters:
        model_names : List[str]
            A list of string identifiers for the models whose scores are to be loaded.
            These names should correspond to how the scores are organized on disk.
        dir : Optional[str], default None
            An optional subdirectory name within the scores' base directory.
            This is useful for organizing scores by different validation sets
            (e.g., 'val', 'high_intensity').
        leadtimes : List[int], default [5, 10, 15, 30, 60, 90]
            A list of lead times (in minutes) for which scores are expected.
            These are passed to the `Evaluator` to filter relevant scores.
        thresholds : List[float], default [0.5, 1, 2, 5, 10, 20, 30]
            A list of rainfall intensity thresholds (in mm/h) for which scores
            are expected (especially for categorical metrics). These are passed
            to the `Evaluator`.

    Returns:
        Tuple[Dict, Dict, Dict]
            A tuple containing three dictionaries:
            - `all_cat`: Dictionary mapping model names to a list of categorical score dictionaries.
                         Each score dictionary contains 'leadtime', 'threshold', and metric values (e.g., 'POD', 'CSI').
            - `all_cont`: Dictionary mapping model names to a list of continuous score dictionaries.
                          Each score dictionary contains 'leadtime' and metric values (e.g., 'MSE', 'MAE').
            - `all_fss`: Dictionary mapping model names to a list of FSS score dictionaries.
                         Each score dictionary contains 'leadtime', 'threshold', 'scale', and 'FSS' value.
    """
    all_cat: Dict[str, List[Dict[str, Any]]] = {}
    all_cont: Dict[str, List[Dict[str, Any]]] = {}
    all_fss: Dict[str, List[Dict[str, Any]]] = {}

    for m in model_names:
        print(f"Loading scores for model: {m}")
        # Initialize the Evaluator for the current model.
        # The 'dir' parameter to Evaluator should point to the base directory
        # where the model's score files are located.
        ev = Evaluator(nowcast_method=m, dir=dir, leadtimes=leadtimes, thresholds=thresholds)
        
        # Load the accumulated scores from disk.
        ev.load_accum_scores()
        
        # Retrieve the parsed scores.
        cat, cont, fss = ev.get_scores()
        
        # Store the scores, keyed by model name.
        all_cat[m] = cat
        all_cont[m] = cont
        all_fss[m] = fss
    return all_cat, all_cont, all_fss

def plot_categorical_scores(all_cat_scores: Dict[str, List[Dict[str, Any]]], merge_mode: str = "by_threshold"):
    """
    Generates and saves plots for categorical verification metrics.

    Plots compare different models across various leadtimes and thresholds
    based on the specified `merge_mode`. Supported metrics include POD, CSI, FAR, and BIAS.

    Parameters:
        all_cat_scores : Dict[str, List[Dict[str, Any]]]
            A dictionary containing categorical scores for all models, as returned
            by `load_model_scores`.
        merge_mode : str, default "by_threshold"
            Determines how the plots are organized:
            - "by_threshold": Creates one subplot for each leadtime. The X-axis
              of each subplot represents different thresholds.
            - "by_leadtimes": Creates one subplot for each threshold. The X-axis
              of each subplot represents different leadtimes.
            - "avg_leadtimes": Creates a single plot where metrics are averaged
              across all leadtimes. The X-axis represents thresholds.
            - "avg_thresholds": Creates a single plot where metrics are averaged
              across all thresholds. The X-axis represents leadtimes.

    Raises:
        ValueError: If an unknown `merge_mode` is provided.
    """
    # Extract unique thresholds and leadtimes present in the loaded scores.
    # This ensures the plots adapt to the actual data available.
    thresholds = sorted(list({s['threshold'] for scores in all_cat_scores.values() for s in scores}))
    leadtimes  = sorted(list({s['leadtime']  for scores in all_cat_scores.values() for s in scores}))
    
    # Define the categorical metrics to plot.
    metrics = ["POD", "CSI", "FAR", "BIAS"]

    for metric in metrics:
        if merge_mode == "by_threshold":
            # Plotting: One subplot per leadtime, X-axis is thresholds.
            fig, axs = plt.subplots(1, len(leadtimes), figsize=(5 * len(leadtimes), 4), sharey=True)
            if len(leadtimes) == 1: # Handle case of single leadtime to avoid array indexing issues
                axs = [axs]
            for i, lt in enumerate(leadtimes):
                ax = axs[i]
                for model, scores in all_cat_scores.items():
                    # Filter scores for the current leadtime and extract the metric values.
                    vals = [s[metric] for s in scores if s['leadtime'] == lt]
                    ax.plot(thresholds, vals, marker='o', label=model)
                ax.set_title(f"LT={lt} min")
                ax.set_xlabel("Threshold (mm/h)")
                if i == 0: # Only set Y-label for the first subplot
                    ax.set_ylabel(metric)
                ax.grid(True)
            axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the last subplot
            fname = f"{metric.lower()}_by_threshold.png"

        elif merge_mode == "by_leadtimes":
            # Plotting: One subplot per threshold, X-axis is leadtimes.
            fig, axs = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 4), sharey=True)
            if len(thresholds) == 1: # Handle case of single threshold
                axs = [axs]
            for j, thr in enumerate(thresholds):
                ax = axs[j]
                for model, scores in all_cat_scores.items():
                    # Filter scores for the current threshold and extract the metric values.
                    vals = [s[metric] for s in scores if s['threshold'] == thr]
                    ax.plot(leadtimes, vals, marker='o', label=model)
                ax.set_title(f"Thr={thr} mm/h")
                ax.set_xlabel("Leadtime (min)")
                if j == 0: # Only set Y-label for the first subplot
                    ax.set_ylabel(metric)
                ax.grid(True)
            axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fname = f"{metric.lower()}_by_leadtimes.png"

        elif merge_mode == "avg_leadtimes":
            # Plotting: Single plot, X-axis is thresholds, averaged over leadtimes.
            plt.figure(figsize=(6, 4))
            for model, scores in all_cat_scores.items():
                vals = [
                    np.mean([s[metric] for s in scores if s['threshold'] == thr])
                    for thr in thresholds
                ]
                plt.plot(thresholds, vals, marker='o', label=model)
            plt.title(f"Avg. {metric} over Leadtimes")
            plt.xlabel("Threshold (mm/h)")
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend()
            fname = f"avg_{metric.lower()}_thresholds.png"

        elif merge_mode == "avg_thresholds":
            # Plotting: Single plot, X-axis is leadtimes, averaged over thresholds.
            plt.figure(figsize=(6, 4))
            for model, scores in all_cat_scores.items():
                vals = [
                    np.mean([s[metric] for s in scores if s['leadtime'] == lt])
                    for lt in leadtimes
                ]
                plt.plot(leadtimes, vals, marker='o', label=model)
            plt.title(f"Avg. {metric} over Thresholds")
            plt.xlabel("Leadtime (min)")
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend()
            fname = f"avg_{metric.lower()}_leadtimes.png"

        else:
            raise ValueError(f"Unknown merge_mode={merge_mode}. Must be one of 'by_threshold', 'by_leadtimes', 'avg_leadtimes', 'avg_thresholds'.")

        plt.tight_layout() # Adjust layout to prevent labels/titles from overlapping
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=150) # Save figure with specified DPI
        plt.close() # Close the figure to free up memory

def plot_continuous_scores(all_cont_scores: Dict[str, List[Dict[str, Any]]]):
    """
    Generates and saves plots for continuous verification metrics (MSE, MAE).

    These metrics are typically plotted against leadtime.

    Parameters:
        all_cont_scores : Dict[str, List[Dict[str, Any]]]
            A dictionary containing continuous scores for all models, as returned
            by `load_model_scores`.
    """
    # Extract unique leadtimes from the scores.
    leadtimes = sorted(list({s['leadtime'] for scores in all_cont_scores.values() for s in scores}))
    
    # Define the continuous metrics to plot.
    metrics = ["MSE", "MAE"]

    for metric in metrics:
        plt.figure(figsize=(6, 4)) # Create a new figure for each metric
        for model, scores in all_cont_scores.items():
            # Extract metric values for the current model.
            vals = [s[metric] for s in scores]
            plt.plot(leadtimes, vals, marker='o', label=model)
        plt.title(f"{metric} vs Leadtime")
        plt.xlabel("Leadtime (min)")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        fname = f"{metric.lower()}_by_leadtime.png"
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
        plt.close()

def plot_fss_scores(all_fss_scores: Dict[str, List[Dict[str, Any]]]):
    """
    Generates and saves plots for the Fractions Skill Score (FSS).

    FSS is a scale-dependent metric, so plots typically show FSS vs. spatial scale,
    leadtime, or threshold. This function generates several common FSS plots.

    Parameters:
        all_fss_scores : Dict[str, List[Dict[str, Any]]]
            A dictionary containing FSS scores for all models, as returned
            by `load_model_scores`.
    """
    # Extract unique leadtimes, thresholds, and scales present in the loaded scores.
    leadtimes = sorted(list({s['leadtime'] for scores in all_fss_scores.values() for s in scores}))
    thresholds = sorted(list({s['threshold'] for scores in all_fss_scores.values() for s in scores}))
    scales = sorted(list({s['scale'] for scores in all_fss_scores.values() for s in scores}))
    
    # --- Plot 1: FSS vs Spatial Scale (for fixed threshold and leadtime) ---
    # These fixed values are hardcoded for demonstration. In a production system,
    # these might be configurable or automatically selected (e.g., median values).
    fixed_threshold_1 = 1.0   # mm/h
    fixed_leadtime_1 = 30     # min
    
    # TODO: [Refinement] Add checks here to ensure `fixed_threshold_1` and `fixed_leadtime_1`
    # actually exist in the `thresholds` and `leadtimes` lists, respectively.
    # If not, provide a warning or select the closest available value.

    plt.figure(figsize=(6, 4))
    for model, scores in all_fss_scores.items():
        vals = [s['FSS'] for s in scores 
                if s['threshold'] == fixed_threshold_1 
                and s['leadtime'] == fixed_leadtime_1]
        plt.plot(scales, vals, marker='o', label=model)
    plt.xlabel('Spatial Scale (km)')
    plt.ylabel('FSS')
    plt.title(f'FSS at Thr={fixed_threshold_1}mm/h, LT={fixed_leadtime_1}min')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fss_vs_scale_thr{fixed_threshold_1}_lt{fixed_leadtime_1}.png"), dpi=150)
    plt.close()
    
    # --- Plot 2: FSS vs Leadtime (for fixed threshold and scale) ---
    fixed_threshold_2 = 1.0   # mm/h
    fixed_scale_2 = 16        # km
    
    # TODO: [Refinement] Add checks here to ensure `fixed_threshold_2` and `fixed_scale_2`
    # exist in the `thresholds` and `scales` lists.

    plt.figure(figsize=(6, 4))
    for model, scores in all_fss_scores.items():
        vals = [s['FSS'] for s in scores 
                if s['threshold'] == fixed_threshold_2 
                and s['scale'] == fixed_scale_2]
        plt.plot(leadtimes, vals, marker='o', label=model)
    plt.xlabel('Leadtime (min)')
    plt.ylabel('FSS')
    plt.title(f'FSS at Thr={fixed_threshold_2}mm/h, Scale={fixed_scale_2}km')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fss_vs_leadtime_thr{fixed_threshold_2}_scale{fixed_scale_2}.png"), dpi=150)
    plt.close()
    
    # --- Plot 3: FSS vs Threshold (for fixed leadtime and scale) ---
    fixed_leadtime_3 = 30     # min
    fixed_scale_3 = 16        # km

    # TODO: [Refinement] Add checks here to ensure `fixed_leadtime_3` and `fixed_scale_3`
    # exist in the `leadtimes` and `scales` lists.

    plt.figure(figsize=(6, 4))
    for model, scores in all_fss_scores.items():
        vals = [s['FSS'] for s in scores 
                if s['leadtime'] == fixed_leadtime_3 
                and s['scale'] == fixed_scale_3]
        plt.plot(thresholds, vals, marker='o', label=model)
    plt.xlabel('Threshold (mm/h)')
    plt.ylabel('FSS')
    plt.title(f'FSS at LT={fixed_leadtime_3}min, Scale={fixed_scale_3}km')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fss_vs_threshold_lt{fixed_leadtime_3}_scale{fixed_scale_3}.png"), dpi=150)
    plt.close()
    
    # --- Plot 4: FSS vs Spatial Scale (averaged over thresholds and leadtimes) ---
    plt.figure(figsize=(6, 4))
    for model, scores in all_fss_scores.items():
        avg_fss = []
        for scale in scales:
            # Collect all FSS scores for the current scale, regardless of leadtime or threshold.
            scale_scores = [s['FSS'] for s in scores if s['scale'] == scale]
            # Calculate the mean of these scores. Handle empty list if no scores for a scale.
            avg_fss.append(np.mean(scale_scores) if scale_scores else np.nan)
        plt.plot(scales, avg_fss, marker='o', label=model)
    plt.xlabel('Spatial Scale (km)')
    plt.ylabel('Average FSS')
    plt.title('FSS Averaged Over Thresholds & Leadtimes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fss_avg_vs_scale.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    """
    Main execution block for loading model scores and generating plots.

    This block defines which models to compare, the directory where their
    scores are stored, and the leadtimes/thresholds to consider. It then
    calls the score loading and plotting functions.

    To run this script:
    `python scripts/plot_scores.py`

    Before running, ensure:
    1. The `nowcasting.metrics.validation.Evaluator` class is correctly implemented
       and accessible, and it can load score files from the specified `dir`.
    2. Score files exist at the expected locations (./results/scores/<dir>/<model>).
    3. The selected models were all evaluated with the same 'leadtimes' and 'thresholds'.

    Note: 
    """
    
    models = [
        'pysteps', 
        # 'ef_rad', 
        # 'ef_rad_balancedloss',
        # 'ef_rad_balancedloss_full',
        # 'ef_rad_balancedextendedloss',
        # 'ef_rad_sat_balancedextendedloss',
        # 'ef_rad_sat', 
        # 'ef_rad_sat_resample_low', 
        # 'ef_rad_sat_resample_high'
        'ldcast_nowcast_balancedext',
        'ldcast_nowcast_sat_pretrained'
    ]
    
    dir = 'val' #'high_intensity'
    leadtimes = [5,10,15,30,60,90]
    thresholds = [0.5,1,2,5,10,20,30]

    if dir:
        FIG_DIR = f'{FIG_DIR}/{dir}'
        os.makedirs(FIG_DIR, exist_ok=True)

    all_cat, all_cont, all_fss = load_model_scores(models, dir=dir,leadtimes=leadtimes, thresholds=thresholds)

    for merge_mode in ["by_threshold", "by_leadtimes", "avg_leadtimes", "avg_thresholds"]:
        plot_categorical_scores(all_cat, merge_mode=merge_mode)
    plot_continuous_scores(all_cont)
    plot_fss_scores(all_fss)
    print(f"Saved plots to {FIG_DIR}")


    # TODO: Refactor plotting parameters to be loaded from a config file (e.g., YAML).
    # This involves:
    # 1.  Using a configuration library (like OmegaConf) to load all plotting settings
    #     (models, leadtimes, thresholds, output paths, and FSS-specific fixed values).
    # 2.  Updating functions to accept these parameters dynamically.
    # 3.  Implementing robust handling for FSS fixed parameters: ensure chosen values
    #     exist in the loaded data, or provide clear fallback/error mechanisms
    #     (e.g., automatically pick nearest, warn, or error if not found).
    # TODO: Check that all models have the same leadtimes and thresholds