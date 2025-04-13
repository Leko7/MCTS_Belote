# Utils.py
import matplotlib.pyplot as plt

def save_csv(df_sorted, csv_path):
    # Save as CSV
    df_sorted.to_csv(csv_path, index=False)

def save_png(df_sorted, png_path):

    # Compute size based on data
    n_rows, n_cols = df_sorted.shape
    fig_width = n_cols * 1.5
    fig_height = n_rows * 0.6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(cellText=df_sorted.values,
                     colLabels=df_sorted.columns,
                     cellLoc='center',
                     loc='center')

    # Manually set font size to be larger
    table.auto_set_font_size(False)
    table.set_fontsize(3)

    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)