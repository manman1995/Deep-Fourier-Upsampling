import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def extract_epoch_psnr_from_log(log_file):
    epochs = []
    psnr_values = []

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            current_epoch = None

            for line in lines:
                epoch_match = re.search(r'\[epoch:\s*(\d+),', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))

                psnr_match = re.search(r'Validation RAIN200H,\s+# psnr:\s*(\d+\.\d+)', line)
                if psnr_match and current_epoch is not None:
                    psnr = float(psnr_match.group(1))
                    epochs.append(current_epoch)
                    psnr_values.append(psnr)

        return epochs, psnr_values
    except Exception as e:
        print(f"Error processing file {log_file}: {e}")
        return [], []

def save_max_psnr_info(log_files, labels=None, output_path="max_psnr_results.txt", scale_flags=None):
    results = []
    
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(log_files))]
    
    for i, log_file in enumerate(log_files):
        epochs, psnr_values = extract_epoch_psnr_from_log(log_file)
        psnr_values = [v * (scale_flags[i] if scale_flags else 1.0) for v in psnr_values]
        
        if psnr_values:
            max_index = np.argmax(psnr_values)
            max_psnr = psnr_values[max_index]
            best_epoch = epochs[max_index]
            results.append(f"{labels[i]}: Max PSNR = {max_psnr:.4f} at Epoch {best_epoch}")
        else:
            results.append(f"{labels[i]}: No PSNR data found.")
    
    with open(output_path, 'w') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"Max PSNR results saved to {output_path}")



def plot_psnr(log_files, labels=None, output_path=None, title="PSNR vs Epochs", map_flags=None, scale_flags=None):
    plt.figure(figsize=(10, 6))
    
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(log_files))]
    
    colors = [
        '#A6CEE3', '#FB9A99', '#B2DF8A', '#FDBF6F',
        '#CAB2D6', '#FFFF99', '#1F78B4', '#33A02C'
    ]
    
    for i, log_file in enumerate(log_files):
        epochs, psnr_values = extract_epoch_psnr_from_log(log_file)

        psnr_values = [v * (scale_flags[i] if scale_flags else 1.0) for v in psnr_values]


        if epochs and psnr_values:
            color = colors[i % len(colors)]
            plt.plot(epochs, psnr_values, linestyle='-', color=color,
                     label=labels[i], linewidth=2)
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xlim(0, 925)

    plt.xticks(np.linspace(0, 925, 10, dtype=int), rotation=45)

    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log files and plot PSNR vs Epoch')
    parser.add_argument('log_files', nargs='+', help='Paths to the log files')
    parser.add_argument('--labels', nargs='+', help='Labels for each log file')
    parser.add_argument('--output', help='Path to save the output plot')
    parser.add_argument('--title', default='PSNR vs Epochs', help='Title for the plot')
    parser.add_argument('--scale_flags', nargs='+', type=float, help='Scaling factors for each log (e.g., 1.0 0.95)')
    parser.add_argument('--psnr_info_output', default='max_psnr_results.txt', help='Output file for max PSNR info')


    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.log_files):
        print("Warning: Number of labels does not match number of log files")
        args.labels = None
    
    if args.scale_flags and len(args.scale_flags) != len(args.log_files):
        print("Warning: Number of scale flags does not match number of log files")
        args.scale_flags = [1.0] * len(args.log_files)
    elif not args.scale_flags:
        args.scale_flags = [1.0] * len(args.log_files)


    plot_psnr(args.log_files, args.labels, args.output, args.title, scale_flags=args.scale_flags)
    save_max_psnr_info(args.log_files, args.labels, args.psnr_info_output, scale_flags=args.scale_flags)
