import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import ast
import os
import numpy as np # For log scale handling
from matplotlib import ticker as mticker
from scipy.optimize import curve_fit

# --- Configuration ---
LOG_DIR = '.' # Directory containing the log files
OUTPUT_DIR = 'plots_per_run' # Directory to save plots for each run

# --- Helper Functions ---

def parse_num_agents_from_filename(filename, prefix):
    """Extracts the number of agents from a filename."""
    # Regex to capture digits after '_agents_' and before any other non-digit character or end
    base_name = os.path.basename(filename)
    match = re.search(rf'{prefix}_agents_(\d+)[_\s.]', base_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
             print(f"Warning: Found non-integer value after '_agents_' in {base_name}")
             return None
    else:
        # Fallback: try matching if number is at the very end before .csv
        match = re.search(rf'{prefix}_agents_(\d+)\.csv$', base_name)
        if match:
             try:
                 return int(match.group(1))
             except ValueError:
                 print(f"Warning: Found non-integer value at end in {base_name}")
                 return None
        print(f"Warning: Could not parse number of agents from filename: {base_name}")
        return None

def load_task_completion_logs(log_dir, prefix='task_completion_log'):
    """Loads all task completion logs into a single DataFrame."""
    all_files = glob.glob(os.path.join(log_dir, f'{prefix}_agents_*.csv'))
    print(f"Found task completion files: {all_files}")
    df_list = []
    for f in all_files:
        num_agents = parse_num_agents_from_filename(f, prefix)
        if num_agents is not None:
            print(f"Processing task file {os.path.basename(f)} for {num_agents} agents...")
            try:
                df = pd.read_csv(f, header=None, names=['agent_id', 'task_id', 'completion_time'], on_bad_lines='warn')
                df['agent_id'] = pd.to_numeric(df['agent_id'], errors='coerce')
                df['task_id'] = pd.to_numeric(df['task_id'], errors='coerce')
                df['completion_time'] = pd.to_numeric(df['completion_time'], errors='coerce')
                initial_rows = len(df)
                df.dropna(inplace=True)
                if initial_rows > len(df):
                    print(f"  Dropped {initial_rows - len(df)} rows with NaN values from {os.path.basename(f)}")
                if not df.empty:
                    df['agent_id'] = df['agent_id'].astype(int)
                    df['task_id'] = df['task_id'].astype(int)
                    df['num_agents'] = num_agents
                    df_list.append(df)
                else:
                    print(f"  No valid data rows left in {os.path.basename(f)} after cleaning.")
            except Exception as e:
                print(f"Error processing file {f}: {e}")
        else:
             print(f"Skipping task file (could not determine agent count): {f}")
    if not df_list:
        print("No task completion dataframes were created.")
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def load_optimization_logs(log_dir, prefix='optimization_log'):
    """Loads all optimization logs into a single DataFrame."""
    all_files = glob.glob(os.path.join(log_dir, f'{prefix}_agents_*.csv'))
    print(f"Found optimization files: {all_files}")
    df_list = []
    for f in all_files:
        num_agents = parse_num_agents_from_filename(f, prefix)
        if num_agents is not None:
            print(f"Processing optimization file {os.path.basename(f)} for {num_agents} agents...")
            try:
                data = []
                line_count = 0
                skipped_lines = 0
                with open(f, 'r') as infile:
                    for line in infile:
                        line_count += 1
                        line = line.strip()
                        # Skip empty lines and lines that look like headers
                        if not line or ('agent_id' in line and 'optimization_time' in line and 'iterations' in line):
                             skipped_lines +=1
                             continue

                        parts = line.split(',', 3)
                        if len(parts) == 4:
                             try:
                                 agent_id = int(parts[0])
                                 opt_time = float(parts[1])
                                 iterations = int(parts[2])
                                 tasks_str = parts[3].strip().strip('"') # Handle potential quotes
                                 if tasks_str.startswith('[') and tasks_str.endswith(']'):
                                     tasks = ast.literal_eval(tasks_str)
                                     if not isinstance(tasks, list): # Ensure it's a list
                                         tasks = []
                                 else:
                                     tasks = [] # Assign empty list if not in list format
                                 data.append([agent_id, opt_time, iterations, tasks, len(tasks)])
                             except (ValueError, SyntaxError, TypeError) as parse_err:
                                 print(f"  Skipping malformed data line {line_count} in {os.path.basename(f)}: {line} | Error: {parse_err}")
                                 skipped_lines += 1
                                 continue
                        else:
                             print(f"  Skipping line {line_count} with unexpected number of columns ({len(parts)}) in {os.path.basename(f)}: {line}")
                             skipped_lines += 1

                print(f"  Processed {line_count} lines, skipped {skipped_lines} lines in {os.path.basename(f)}.")
                if data:
                    # Aggregate data per file if multiple entries per agent exist
                    temp_df = pd.DataFrame(data, columns=['agent_id', 'optimization_time', 'iterations', 'tasks', 'num_tasks_assigned'])
                    # Keep the last entry for each agent within this file
                    final_data_df = temp_df.drop_duplicates(subset=['agent_id'], keep='last')
                    final_data_df['num_agents'] = num_agents
                    df_list.append(final_data_df)
                else:
                    print(f"  No valid data rows extracted from {os.path.basename(f)}.")
            except Exception as e:
                print(f"Error processing file {f}: {e}")
        else:
             print(f"Skipping optimization file (could not determine agent count): {f}")

    if not df_list:
        print("No optimization dataframes were created.")
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)


# --- Plotting Functions ---

# Removed plot_iterations_vs_agent_tasks as requested

def plot_task_completion_timeline_per_run(df_tasks_subset, n_agents, output_dir):
    """Plots task completion timeline for a specific run."""
    plot_name = f'task_completion_timeline_{n_agents}_agents'
    if df_tasks_subset.empty:
        print(f"Skipping plot '{plot_name}': No task completion data for {n_agents} agents.")
        return

    print(f"Plotting '{plot_name}'...")
    df_plot = df_tasks_subset.sort_values('completion_time')

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_plot, x='completion_time', y='task_id', hue='agent_id', palette='viridis', s=100, legend='full')
    plt.title(f'Task Completion Timeline ({n_agents} Agents)')
    plt.xlabel('Completion Time (s)')
    plt.ylabel('Task ID')

    # Adjust legend position
    if len(df_plot['agent_id'].unique()) > 10:
        plt.legend(title='Agent ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make space for legend
    else:
        plt.legend(title='Agent ID')
        plt.tight_layout()

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    filename = os.path.join(output_dir, f'{plot_name}.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def plot_tasks_per_agent_bar_chart(df_opt_subset, n_agents, output_dir):
    """Plots bar chart of tasks assigned per agent for a specific run."""
    plot_name = f'tasks_per_agent_barchart_{n_agents}_agents'
    if df_opt_subset.empty or 'agent_id' not in df_opt_subset.columns or 'num_tasks_assigned' not in df_opt_subset.columns:
        print(f"Skipping plot '{plot_name}': Missing required optimization data.")
        return

    # Data already filtered to keep last entry per agent in loading function
    df_plot = df_opt_subset.sort_values('agent_id')

    if df_plot.empty or df_plot['num_tasks_assigned'].isnull().all():
        print(f"Skipping plot '{plot_name}': No valid task assignment counts.")
        return

    print(f"Plotting '{plot_name}'...")
    plt.figure(figsize=(max(8, n_agents * 0.6), 6)) # Adjust width based on number of agents
    sns.barplot(data=df_plot, x='agent_id', y='num_tasks_assigned', palette='viridis', hue='agent_id', dodge=False, legend=False) # Use hue for color, disable barplot legend

    plt.title(f'Number of Tasks Assigned per Agent ({n_agents} Agents)')
    plt.xlabel('Agent ID')
    plt.ylabel('Number of Tasks Assigned')
    plt.xticks(rotation=0) # Keep labels horizontal unless too many agents
    if n_agents > 15:
         plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout

    filename = os.path.join(output_dir, f'{plot_name}.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def saturation_function(x, C, k):
    """Exponential saturation function: C * (1 - exp(-k*x))"""
    # Ensure k is positive to have saturation behavior
    k = abs(k)
    return C * (1 - np.exp(-k * x))

def plot_avg_iterations_vs_total_tasks(df_opt, output_dir):
    """Plots average iterations vs total tasks assigned, fitting a saturation curve."""
    plot_name = 'avg_iterations_vs_total_tasks_fit' # Changed name slightly
    if df_opt.empty or 'iterations' not in df_opt.columns or 'num_tasks_assigned' not in df_opt.columns or 'num_agents' not in df_opt.columns:
        print(f"Skipping plot '{plot_name}': Missing required optimization data.")
        return

    # Group by the simulation run (num_agents) and calculate aggregates
    summary = df_opt.groupby('num_agents').agg(
        avg_iterations=('iterations', 'mean'),
        total_tasks=('num_tasks_assigned', 'sum')
    ).reset_index()

    summary = summary[summary['total_tasks'] > 0]

    if summary.empty or len(summary) < 2: # Need at least 2 points to fit a curve
        print(f"Skipping plot '{plot_name}': Not enough valid data points for curve fitting.")
        return

    summary = summary.sort_values('total_tasks')

    x_data = summary['total_tasks'].values
    y_data = summary['avg_iterations'].values

    # --- Curve Fitting ---
    try:
        # Provide initial guesses (p0) if possible, C ~ max iterations, k ~ small positive number
        initial_guess = [y_data.max(), 0.01]
        # Set bounds: C must be positive, k must be positive
        bounds = ([0, 0], [np.inf, np.inf])
        
        params, covariance = curve_fit(saturation_function, x_data, y_data, p0=initial_guess, bounds=bounds, maxfev=5000)
        C_fit, k_fit = params
        print(f"Curve fit parameters for {plot_name}: C={C_fit:.4f}, k={k_fit:.4f}")

        # Generate smooth x values for plotting the curve, starting from 0
        x_smooth = np.linspace(0, x_data.max() * 1.1, 200) # Extend slightly beyond max observed x
        y_smooth = saturation_function(x_smooth, C_fit, k_fit)
        fit_successful = True

    except RuntimeError as e:
        print(f"Warning: Curve fitting failed for '{plot_name}': {e}. Plotting only raw data.")
        fit_successful = False
    except Exception as e:
         print(f"Warning: An unexpected error occurred during curve fitting for '{plot_name}': {e}. Plotting only raw data.")
         fit_successful = False
    # --- Plotting ---
    print(f"Plotting '{plot_name}'...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the original data points
    sns.lineplot(data=summary, x='total_tasks', y='avg_iterations', ax=ax, label='Observed Data', zorder=5)

    # Plot the fitted curve if successful
    if fit_successful:
        ax.plot(x_smooth, y_smooth, color='red', label=f'Fit: C*(1-exp(-k*x))\nC={C_fit:.2f}, k={k_fit:.3f}', zorder=4)
        # Optionally, add a horizontal line for the asymptote C
        ax.axhline(y=C_fit, color='red', linestyle=':', alpha=0.7, label=f'Asymptote (C={C_fit:.2f})')


    ax.set_title('Average CBBA Iterations vs. Total Tasks (Saturation Fit)')
    ax.set_xlabel('Total Number of Tasks Assigned in Simulation')
    ax.set_ylabel('Average Iterations')

    # Set axes limits starting from 0
    ax.set_xlim(left=0)
    min_y_limit = 0 # Start y-axis at 0
    max_y_limit = max(y_data.max() * 1.1, C_fit * 1.1 if fit_successful else y_data.max() * 1.1) # Adjust upper limit based on data/fit
    ax.set_ylim(bottom=min_y_limit, top=max_y_limit)


    # Set y-axis to use integer ticks if appropriate range
    # Use MaxNLocator for automatic integer ticks if the range allows
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=4))


    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()

    filename = os.path.join(output_dir, f'{plot_name}.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close(fig)




# --- Main Execution ---
if __name__ == "__main__":
    print(f"Log directory: {os.path.abspath(LOG_DIR)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Load data
    print("\n--- Loading Task Completion Logs ---")
    df_tasks = load_task_completion_logs(LOG_DIR)
    print(f"Loaded {len(df_tasks)} total rows into df_tasks.")

    print("\n--- Loading Optimization Logs ---")
    df_opt = load_optimization_logs(LOG_DIR)
    print(f"Loaded {len(df_opt)} total rows into df_opt.")

    if df_tasks.empty and df_opt.empty:
        print("\nNo valid log data parsed from any files. Exiting.")
    else:
        # Get unique numbers of agents found in logs
        agents_in_tasks = []
        if not df_tasks.empty and 'num_agents' in df_tasks.columns:
             agents_in_tasks = df_tasks['num_agents'].dropna().unique()
             agents_in_tasks = [int(a) for a in agents_in_tasks]

        agents_in_opt = []
        if not df_opt.empty and 'num_agents' in df_opt.columns:
             agents_in_opt = df_opt['num_agents'].dropna().unique()
             agents_in_opt = [int(a) for a in agents_in_opt]

        all_num_agents = sorted(list(set(agents_in_tasks) | set(agents_in_opt)))

        print("\n--- Data Loading Summary ---")
        print(f"Task completion agent counts found: {sorted(agents_in_tasks) if agents_in_tasks else 'None'}")
        print(f"Optimization agent counts found: {sorted(agents_in_opt) if agents_in_opt else 'None'}")
        print(f"Overall agent counts for plotting: {all_num_agents if all_num_agents else 'None'}")
        print("--------------------------\n")

        # Generate summary plot across runs
        plot_avg_iterations_vs_total_tasks(df_opt, OUTPUT_DIR)

        # Generate plots per run
        if not all_num_agents:
            print("No agent data found to generate per-run plots.")
        else:
            print("Generating plots for each simulation run...")
            for n_agents in all_num_agents:
                print(f"\n--- Generating plots for {n_agents} agents ---")
                # Filter data for the current number of agents
                df_tasks_subset = df_tasks[df_tasks['num_agents'] == n_agents] if not df_tasks.empty else pd.DataFrame()
                df_opt_subset = df_opt[df_opt['num_agents'] == n_agents] if not df_opt.empty else pd.DataFrame()

                # Call plotting functions for this subset (excluding the removed one)
                plot_task_completion_timeline_per_run(df_tasks_subset, n_agents, OUTPUT_DIR)
                plot_tasks_per_agent_bar_chart(df_opt_subset, n_agents, OUTPUT_DIR)

            print("\nVisualization complete. Plots saved in:", os.path.abspath(OUTPUT_DIR))
