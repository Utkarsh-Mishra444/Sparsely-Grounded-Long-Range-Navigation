#!/usr/bin/env python3
"""
Batch process all runs in a dataset to recalculate decision evaluations.

Usage:
    python recalculate_all_runs.py <dataset_folder> [--recomputed-only]

Options:
    --recomputed-only    Only show average per-run accuracy for recomputed evaluations

Examples:
    python recalculate_all_runs.py "logs/100_paths_full_trapi_gpt-4o/20250918_001718_ad9f1a42"
    python recalculate_all_runs.py "logs/100_paths_full_trapi_gpt-5" --recomputed-only
"""

import json
import os
import sys
import glob
import argparse
from datetime import datetime
from analysis.recalculate.evaluations import (
    load_run_data,
    recalculate_evaluations,
    reset_walking_distance_stats,
    get_walking_distance_api_calls,
)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def find_all_run_folders(dataset_folder):
    """Find all run folders in the dataset, traversing nested timestamp directories."""
    run_folders = []

    for root, dirnames, _ in os.walk(dataset_folder):
        for dirname in dirnames:
            if not (dirname.startswith("gpt") or dirname.startswith("gemini") or "ollama" in dirname):
                continue
            folder = os.path.join(root, dirname)
            if not os.path.isdir(folder):
                continue

            # Check for files in this folder and all subfolders
            has_coords = False
            has_log = False
            actual_run_folder = folder

            for root, _, files in os.walk(folder):
                if "visited_coordinates.json" in files:
                    has_coords = True
                    actual_run_folder = root
                if any(f.startswith("detailed_decision_log_") and f.endswith(".json") for f in files):
                    has_log = True
                    actual_run_folder = root
                if has_coords and has_log:
                    break

            print(f"Checking folder: {folder}")
            print(f"  has_coords: {has_coords}, has_log: {has_log}")
            print(f"  actual_run_folder: {actual_run_folder}")

            if has_coords or has_log:
                run_folders.append(actual_run_folder)

    print(f"Found {len(run_folders)} run folders total")
    return sorted(run_folders)


def load_ground_truth(run_folder):
    eval_files = sorted(glob.glob(os.path.join(run_folder, "decision_evaluations_*.json")))
    gt_files = [path for path in eval_files if "recalculated" not in os.path.basename(path)]
    if not gt_files:
        raise FileNotFoundError("decision_evaluations_*.json not found")
    with open(gt_files[0], 'r') as f:
        return json.load(f)


def process_single_run(run_folder, run_index, total_runs):
    """Process a single run and return results."""
    # Extract path number from folder name
    path_parts = run_folder.split(os.sep)
    path_id = next((p for p in path_parts if p.startswith("000") and "_path_" in p), "unknown")

    print(f"\n[{run_index+1}/{total_runs}] Processing {path_id}...")

    try:
        ground_truth = load_ground_truth(run_folder)
    except Exception as e:
        print(f"  ⚠ Could not load ground truth: {e}")
        return {
            'run_folder': run_folder,
            'path_id': path_id,
            'status': 'missing_ground_truth',
            'gt_total': 0,
            'gt_right': 0,
            'gt_wrong': 0,
            'recalc_available': False,
            'recalc_total': 0,
            'recalc_right': 0,
            'recalc_wrong': 0,
            'matches': 0,
            'mismatches': 0,
            'accuracy': 0.0,
            'comparisons': []
        }

    gt_total = len(ground_truth)
    gt_right = sum(1 for entry in ground_truth if entry.get('status') == 'RIGHT')
    gt_wrong = gt_total - gt_right

    # Check if run was successful (reached destination)
    is_successful = False
    if ground_truth and isinstance(ground_truth[-1], dict):
        is_successful = ground_truth[-1].get('reached_destination', False)

    gt_accuracy = (gt_right / gt_total * 100) if gt_total else 0

    try:
        visited_coordinates, detailed_decision_log, destination_coords = load_run_data(run_folder)
        recalculated_evals = recalculate_evaluations(
            visited_coordinates,
            detailed_decision_log,
            destination_coords,
            run_folder
        )
        recalc_available = True
    except Exception as e:
        print(f"  ⚠ Could not recalculate evaluations: {e}")
        recalculated_evals = []
        recalc_available = False

    recalc_total = len(recalculated_evals)
    recalc_right = sum(1 for entry in recalculated_evals if entry.get('status') == 'RIGHT')
    recalc_wrong = recalc_total - recalc_right

    matches = 0
    mismatches = 0
    comparisons = []

    if recalc_available:
        recalc_map = {entry['decision_number']: entry for entry in recalculated_evals}
        for gt_entry in ground_truth:
            decision_num = gt_entry['decision_number']
            recalc_entry = recalc_map.get(decision_num)
            if recalc_entry is None:
                continue
            match = gt_entry['status'] == recalc_entry['status']
            if match:
                matches += 1
            else:
                mismatches += 1
            comparisons.append({
                'decision_num': decision_num,
                'gt_status': gt_entry['status'],
                'recalc_status': recalc_entry['status'],
                'match': match,
                'distance_change': recalc_entry.get('distance_change')
            })

        accuracy = (
            (matches / (matches + mismatches) * 100)
            if (matches + mismatches) > 0 else 0
        )

        # Save recalculated evaluations without interfering with interface stats
        output_file = os.path.join(run_folder, "decision_evaluations_recalculated.json")

        try:
            with open(output_file, 'w') as f:
                json.dump(recalculated_evals, f, indent=2)
        except Exception as e:
            print(f"  ⚠ Failed to save recalculated evaluations: {e}")

        print(f"  ✓ {matches}/{matches + mismatches} correct ({accuracy:.1f}%)")
    else:
        accuracy = 0.0
        print("  ⚠ Skipping comparison due to missing recalculation data")

    return {
        'run_folder': run_folder,
        'path_id': path_id,
        'status': 'ok' if recalc_available else 'recalc_failed',
        'gt_total': gt_total,
        'gt_right': gt_right,
        'gt_wrong': gt_wrong,
        'gt_accuracy': gt_accuracy,
        'is_successful': is_successful,
        'recalc_available': recalc_available,
        'recalc_total': recalc_total,
        'recalc_right': recalc_right,
        'recalc_wrong': recalc_wrong,
        'matches': matches,
        'mismatches': mismatches,
        'accuracy': accuracy,
        'comparisons': comparisons
    }


def print_summary(results, avg_per_run_accuracy, recomputed_only=False):
    if recomputed_only:
        print("\n" + "="*100)
        print("RECOMPUTED ACCURACY - AVERAGE PER RUN")
        print("="*100)
        print(f"Average recomputed decision accuracy per run: {avg_per_run_accuracy:.2f}%")
        print("="*100 + "\n")
        return

    # Normal full summary
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    gt_total = sum(r['gt_total'] for r in results)
    gt_right = sum(r['gt_right'] for r in results)
    gt_wrong = sum(r['gt_wrong'] for r in results)

    recalc_total = sum(r['recalc_total'] for r in results if r['recalc_available'])
    recalc_right = sum(r['recalc_right'] for r in results if r['recalc_available'])
    recalc_wrong = sum(r['recalc_wrong'] for r in results if r['recalc_available'])

    matches = sum(r['matches'] for r in results if r['recalc_available'])
    mismatches = sum(r['mismatches'] for r in results if r['recalc_available'])

    print(f"\nRuns discovered: {len(results)}")
    print(f"Runs with ground truth: {sum(1 for r in results if r['gt_total'] > 0)}")
    print(f"Runs recalculated successfully: {sum(1 for r in results if r['recalc_available'])}")

    print(f"\nGT decisions: {gt_total} (RIGHT {gt_right}, WRONG {gt_wrong})")
    gt_accuracy = (gt_right / gt_total * 100) if gt_total else 0
    print(f"GT decision accuracy (aggregated across all runs): {gt_accuracy:.2f}%")
    print(f"GT decision accuracy (average of per-run accuracies): {avg_per_run_accuracy:.2f}%")

    if recalc_total:
        print(f"Recomputed decisions: {recalc_total} (RIGHT {recalc_right}, WRONG {recalc_wrong})")
        recalc_accuracy = (recalc_right / recalc_total * 100)
        print(f"Recomputed decision accuracy: {recalc_accuracy:.2f}%")
        overall_accuracy = (matches / (matches + mismatches) * 100) if (matches + mismatches) else 0
        print(f"Agreement with GT: {matches}/{matches + mismatches} ({overall_accuracy:.2f}%)")
    else:
        print("Recomputed decisions: 0")

    print("="*100 + "\n")
    print(f"Walking-distance API calls required: {get_walking_distance_api_calls()}")

    # Additional analysis: Decision accuracy for successful runs only
    successful_runs = [r for r in results if r.get('is_successful', False) and r['gt_total'] > 0]

    if successful_runs:
        print("\n" + "="*100)
        print("SUCCESSFUL RUNS ANALYSIS")
        print("="*100)

        successful_accuracies = [r['gt_accuracy'] for r in successful_runs]

        print(f"\nSuccessful runs: {len(successful_runs)}")
        print(f"Mean decision accuracy: {sum(successful_accuracies) / len(successful_accuracies):.2f}%")

        # Create histogram bins (0-10%, 10-20%, ..., 90-100%)
        bins = [(i, i+10) for i in range(0, 100, 10)]
        histogram = {f"{start}-{start+10}%": 0 for start, _ in bins}

        for accuracy in successful_accuracies:
            for start, end in bins:
                if start <= accuracy < end:
                    histogram[f"{start}-{start+10}%"] += 1
                    break
            # Handle 100% case
            if accuracy == 100.0:
                histogram["90-100%"] += 1

        print("\nDecision Accuracy Distribution (Successful Runs):")
        for bin_range in [f"{i}-{i+10}%" for i in range(0, 90, 10)] + ["90-100%"]:
            count = histogram[bin_range]
            percentage = (count / len(successful_runs)) * 100 if successful_runs else 0
            print(f"  {bin_range}: {count} runs ({percentage:.1f}%)")

        print("="*100 + "\n")


def save_histogram_plot(results, output_path):
    """Generate and save histogram plot for successful runs decision accuracy."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping histogram plot generation")
        return

    successful_runs = [r for r in results if r.get('is_successful', False) and r['gt_total'] > 0]
    
    if not successful_runs:
        print("No successful runs to plot")
        return

    successful_accuracies = [r['gt_accuracy'] for r in successful_runs]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create histogram with bins 0-10, 10-20, ..., 90-100
    bins = list(range(0, 110, 10))
    counts, edges, patches = plt.hist(successful_accuracies, bins=bins, edgecolor='black', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Decision Accuracy (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Runs', fontsize=12, fontweight='bold')
    plt.title(f'Decision Accuracy Distribution for Successful Runs (N={len(successful_runs)})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, (count, edge) in enumerate(zip(counts, edges[:-1])):
        if count > 0:
            plt.text(edge + 5, count, str(int(count)), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Set x-axis ticks
    plt.xticks(bins)
    
    # Add mean line
    mean_acc = sum(successful_accuracies) / len(successful_accuracies)
    plt.axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.1f}%')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram plot saved to: {output_path}")


def save_summary_txt(results, output_path, avg_per_run_accuracy, recomputed_only=False):
    """Save a simple text summary of key statistics."""
    with open(output_path, 'w') as f:
        if recomputed_only:
            f.write("="*80 + "\n")
            f.write("RECOMPUTED ACCURACY - AVERAGE PER RUN\n")
            f.write("="*80 + "\n\n")
            f.write(f"Average recomputed decision accuracy per run: {avg_per_run_accuracy:.2f}%\n\n")
            f.write("="*80 + "\n")
            return

        # Normal full summary
        gt_total = sum(r['gt_total'] for r in results)
        gt_right = sum(r['gt_right'] for r in results)
        gt_wrong = sum(r['gt_wrong'] for r in results)
        gt_accuracy = (gt_right / gt_total * 100) if gt_total else 0

        recalc_total = sum(r['recalc_total'] for r in results if r['recalc_available'])
        recalc_right = sum(r['recalc_right'] for r in results if r['recalc_available'])
        recalc_wrong = sum(r['recalc_wrong'] for r in results if r['recalc_available'])
        recalc_accuracy = (recalc_right / recalc_total * 100) if recalc_total else 0

        matches = sum(r['matches'] for r in results if r['recalc_available'])
        mismatches = sum(r['mismatches'] for r in results if r['recalc_available'])
        agreement = (matches / (matches + mismatches) * 100) if (matches + mismatches) else 0

        successful_runs = [r for r in results if r.get('is_successful', False) and r['gt_total'] > 0]
        successful_accuracies = [r['gt_accuracy'] for r in successful_runs] if successful_runs else []
        mean_successful_acc = sum(successful_accuracies) / len(successful_accuracies) if successful_accuracies else 0

        f.write("="*80 + "\n")
        f.write("RECALCULATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Runs with ground truth: {sum(1 for r in results if r['gt_total'] > 0)}\n")
        f.write(f"Runs recalculated successfully: {sum(1 for r in results if r['recalc_available'])}\n\n")

        f.write("-"*80 + "\n")
        f.write("GROUND TRUTH (GT) EVALUATIONS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total decisions: {gt_total}\n")
        f.write(f"Correct (RIGHT): {gt_right}\n")
        f.write(f"Wrong (WRONG): {gt_wrong}\n")
        f.write(f"Accuracy (aggregated across all runs): {gt_accuracy:.2f}%\n")
        f.write(f"Accuracy (average of per-run accuracies): {avg_per_run_accuracy:.2f}%\n\n")

        f.write("-"*80 + "\n")
        f.write("RECALCULATED EVALUATIONS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total decisions: {recalc_total}\n")
        f.write(f"Correct (RIGHT): {recalc_right}\n")
        f.write(f"Wrong (WRONG): {recalc_wrong}\n")
        f.write(f"Accuracy: {recalc_accuracy:.2f}%\n\n")

        f.write("-"*80 + "\n")
        f.write("GT vs RECALCULATED AGREEMENT\n")
        f.write("-"*80 + "\n")
        f.write(f"Matches: {matches}/{matches + mismatches}\n")
        f.write(f"Agreement: {agreement:.2f}%\n\n")

        if successful_runs:
            f.write("-"*80 + "\n")
            f.write("SUCCESSFUL RUNS ONLY\n")
            f.write("-"*80 + "\n")
            f.write(f"Successful runs: {len(successful_runs)}\n")
            f.write(f"Mean decision accuracy: {mean_successful_acc:.2f}%\n\n")

            # Histogram
            bins = [(i, i+10) for i in range(0, 100, 10)]
            histogram = {f"{start}-{start+10}%": 0 for start, _ in bins}
            for accuracy in successful_accuracies:
                for start, end in bins:
                    if start <= accuracy < end:
                        histogram[f"{start}-{start+10}%"] += 1
                        break
                if accuracy == 100.0:
                    histogram["90-100%"] += 1

            f.write("Decision Accuracy Distribution:\n")
            for bin_range in [f"{i}-{i+10}%" for i in range(0, 90, 10)] + ["90-100%"]:
                count = histogram[bin_range]
                percentage = (count / len(successful_runs)) * 100 if successful_runs else 0
                f.write(f"  {bin_range}: {count} runs ({percentage:.1f}%)\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Summary text file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch process all runs in a dataset to recalculate decision evaluations.")
    parser.add_argument("dataset_folder", help="Path to the dataset folder containing runs")
    parser.add_argument("--recomputed-only", action="store_true",
                       help="Only show average per-run accuracy for recomputed evaluations")

    args = parser.parse_args()
    dataset_folder = args.dataset_folder

    if not os.path.exists(dataset_folder):
        print(f"ERROR: Dataset folder does not exist: {dataset_folder}")
        sys.exit(1)

    print(f"Finding all run folders in: {dataset_folder}")
    run_folders = find_all_run_folders(dataset_folder)

    reset_walking_distance_stats()
    print(f"Found {len(run_folders)} run folders to process")

    results = []
    for i, run_folder in enumerate(run_folders):
        result = process_single_run(run_folder, i, len(run_folders))
        results.append(result)

    # Calculate average of per-run accuracies
    if args.recomputed_only:
        # For recomputed-only mode, calculate average per-run accuracy from recalculated results
        runs_with_recalc = [r for r in results if r['recalc_available']]
        if runs_with_recalc:
            per_run_recalc_accuracies = []
            for r in runs_with_recalc:
                recalc_accuracy = (r['recalc_right'] / r['recalc_total'] * 100) if r['recalc_total'] > 0 else 0
                per_run_recalc_accuracies.append(recalc_accuracy)
            avg_per_run_accuracy = sum(per_run_recalc_accuracies) / len(per_run_recalc_accuracies)
        else:
            avg_per_run_accuracy = 0
    else:
        # Normal mode: calculate average per-run accuracy from ground truth
        runs_with_gt = [r for r in results if r['gt_total'] > 0]
        per_run_accuracies = [r['gt_accuracy'] for r in runs_with_gt]
        avg_per_run_accuracy = sum(per_run_accuracies) / len(per_run_accuracies) if per_run_accuracies else 0

    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print summary
    print_summary(results, avg_per_run_accuracy, args.recomputed_only)

    # Save histogram plot
    histogram_plot_path = os.path.join(dataset_folder, f"successful_runs_accuracy_histogram_{timestamp}.png")
    save_histogram_plot(results, histogram_plot_path)

    # Save summary text file
    summary_txt_path = os.path.join(dataset_folder, f"recalculation_summary_{timestamp}.txt")
    save_summary_txt(results, summary_txt_path, avg_per_run_accuracy, args.recomputed_only)

    # Calculate successful runs analysis for JSON
    successful_runs = [r for r in results if r.get('is_successful', False) and r['gt_total'] > 0]
    successful_accuracies = [r['gt_accuracy'] for r in successful_runs] if successful_runs else []

    histogram_data = {}
    if successful_runs:
        bins = [(i, i+10) for i in range(0, 100, 10)]
        histogram_data = {f"{start}-{start+10}%": 0 for start, _ in bins}

        for accuracy in successful_accuracies:
            for start, end in bins:
                if start <= accuracy < end:
                    histogram_data[f"{start}-{start+10}%"] += 1
                    break
            if accuracy == 100.0:
                histogram_data["90-100%"] += 1

    # Save detailed results, including runs without ground truth
    output_file = os.path.join(dataset_folder, f"recalculation_summary_{timestamp}.json")

    with open(output_file, 'w') as f:
        if args.recomputed_only:
            summary_data = {
                'mode': 'recomputed_only',
                'total_runs': len(results),
                'runs_recalculated_successfully': sum(1 for r in results if r['recalc_available']),
                'avg_recomputed_accuracy_per_run': avg_per_run_accuracy,
                'runs': results
            }
        else:
            summary_data = {
                'mode': 'full_summary',
                'total_runs': len(results),
                'runs': results,
                'gt_right': sum(r['gt_right'] for r in results),
                'gt_wrong': sum(r['gt_wrong'] for r in results),
                'gt_total': sum(r['gt_total'] for r in results),
                'gt_accuracy_avg_per_run': avg_per_run_accuracy,
                'recalc_right': sum(r['recalc_right'] for r in results if r['recalc_available']),
                'recalc_wrong': sum(r['recalc_wrong'] for r in results if r['recalc_available']),
                'recalc_total': sum(r['recalc_total'] for r in results if r['recalc_available']),
                'matches': sum(r['matches'] for r in results if r['recalc_available']),
                'mismatches': sum(r['mismatches'] for r in results if r['recalc_available']),
                'successful_runs_analysis': {
                    'successful_runs_count': len(successful_runs),
                    'mean_accuracy': sum(successful_accuracies) / len(successful_accuracies) if successful_accuracies else 0,
                    'accuracy_histogram': histogram_data
                }
            }
        json.dump(summary_data, f, indent=2)

    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
