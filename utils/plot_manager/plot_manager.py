import os
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.log_manager.log_manager import *
from utils.common_variables import *

file_name = os.path.splitext(os.path.basename(__file__))[0]


class PlotManager:
    def __init__(self):
        self.lm = LogManager('main')

    def plot_line(self, path, type_ca, x_values, y_values, x_label, y_label, title, filename, marker='o', markersize=3):
        plt.figure()
        plt.plot(x_values, y_values, color=color_dict[type_ca], linestyle='--', label=type_ca,
                 marker=marker, markersize=markersize)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()
        plt.savefig(f"{path}{filename}", dpi=800)
        self.lm.printl(f"{file_name}. __plot_line finish. {filename} saved.")

    def plot_grid_line(self, path_analysis, filename, df, subset_column, x_column, y_column, x_label, y_label, title):
        self.lm.printl(f"{file_name}. plot_grid_line started.")

        layer_list = list(df[subset_column].unique())
        n_layers = len(layer_list)
        ncols = math.ceil(math.sqrt(n_layers))  # Number of columns
        nrows = math.ceil(n_layers / ncols)  # Number of rows

        # Create subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # subset: for instance the type of co-action/layer
        for i, subset in enumerate(layer_list):
            ax = axes[i]
            subset_df = df[df[subset_column] == subset]
            ax.plot(subset_df[x_column], subset_df[y_column], marker='o', label=f'{subset}', markersize=2)
            ax.set_title(f'{title}: {subset}')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend()

        # Remove any empty subplots
        for j in range(n_layers, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f"{path_analysis}{filename}", dpi=800)
        plt.show()
        self.lm.printl(f"{file_name}. plot_grid_line completed. {filename} saved.")

    def plot_histogram(self, path_analysis, type_ca, values, x_label, y_label, title, filename):
        self.lm.printl(f"{file_name}. plot_histogram started.")
        # Plotting the distribution
        plt.figure()
        plt.hist(values, bins=30, edgecolor='black', color=color_dict[type_ca], label=type_ca, alpha=0.2)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        # Save the figure with high resolution (300 dpi)
        plt.savefig(f"{path_analysis}{filename}", dpi=800)
        plt.show()
        self.lm.printl(f"{file_name}. plot_histogram completed. {filename} saved.")

    def plot_grid_combinations(self, df, path, filename, column1, column2, x_column, y_column, x_label, y_label, step):
        self.lm.printl(f"{file_name}: plot_grid_combinations start.")

        # Get unique actions
        unique_actions1 = df[column1].unique()
        unique_actions2 = df[column2].unique()

        # Create subplots
        fig, axes = plt.subplots(len(unique_actions1), len(unique_actions2), figsize=(15, 15))

        # Plot each pair
        for i, coAction1 in enumerate(unique_actions1):
            for j, coAction2 in enumerate(unique_actions2):
                if i <= j:  # Upper diagonal condition
                    ax = axes[i, j]
                    subset = df[(df[column1] == coAction1) & (df[column2] == coAction2)]
                    if not subset.empty:
                        ax.plot(subset[x_column], subset[y_column], marker='o', markersize=2,
                                label=f'{coAction1} \n {coAction2}')
                        #                 ax.set_title(f'{coAction1} vs {coAction2}')
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_xticks(np.arange(min(subset['threshold']), max(subset['threshold'])+step, step))
                        # ax.set_xlim(0, 0.105)
                        ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels
                        ax.legend()
                        ax.grid(True)

                    else:
                        ax.set_visible(False)  # Hide the subplot if no data
                else:
                    axes[i, j].axis('off')  # Hide the lower triangle plots

        # Add column labels
        for ax, col in zip(axes[0], unique_actions2):
            ax.annotate(f'{col}', xy=(0.5, 1), xytext=(0, 10),
                        xycoords='axes fraction', textcoords='offset points',
                        size='x-large', ha='center', va='baseline')

        # Add row labels
        for ax, row in zip(axes[:, 0], unique_actions1):
            ax.annotate(f'{row}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 50, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        size='x-large', ha='right', va='center', rotation=90)

        # Adjust layout to fit annotations
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{path}{filename}", dpi=800)

        self.lm.printl(f"{file_name}: plot_grid_combinations completed.")

