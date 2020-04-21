import matplotlib.pyplot as plt
import os

from .utils import prepare_directory, generate_file_name


def _set_labels(plt, title_string):

    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor("white")


def _set_axes_lengths(plt):
    # Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400) / (400))


def _add_zone_lines(plt):
    # Plot zone lines
    plt.plot([0, 400], [0, 400], ":", c="black")  # Theoretical 45 regression line
    plt.plot([0, 175 / 3], [70, 70], "-", c="black")
    # plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot(
        [175 / 3, 400 / 1.2], [70, 400], "-", c="black"
    )  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400], "-", c="black")
    plt.plot([0, 70], [180, 180], "-", c="black")
    plt.plot([70, 290], [180, 400], "-", c="black")
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot(
        [70, 70], [0, 56], "-", c="black"
    )  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320], "-", c="black")
    plt.plot([180, 180], [0, 70], "-", c="black")
    plt.plot([180, 400], [70, 70], "-", c="black")
    plt.plot([240, 240], [70, 180], "-", c="black")
    plt.plot([240, 400], [180, 180], "-", c="black")
    plt.plot([130, 180], [0, 70], "-", c="black")


def _add_titles(plt):
    # Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)


def _plot_clarke_pie(zone, suffix, window, model):
    path = prepare_directory("Clarke")
    filepath = os.path.join(path, f"{suffix}_Clarke-Pie.png")
    if os.path.exists(filepath):
        return

    labels = ["A", "B", "C", "D", "E"]
    _, ax1 = plt.subplots(figsize=(15, 15))

    ax1.pie(zone, labels=labels, shadow=True, autopct="%1.1f%%")
    ax1.set_title(f"Clarke_Pie_{suffix}")

    plt.savefig(filepath)
    plt.close()


def _plot_clarke_error(ref_values, pred_values, suffix):
    """This function takes in the reference values and the prediction values as lists 
    and returns a list with each index corresponding to the total number of points 
    within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
    """

    path = prepare_directory("Clarke")
    filepath = os.path.join(path, f"{suffix}_Clarke-Error.png")
    if os.path.exists(filepath):
        return

    # # Clear plot
    plt.clf()
    _, _ = plt.subplots(figsize=(15, 15))
    # Set up plot
    plt.scatter(ref_values, pred_values, marker="o", color="black", s=8)

    title_string = f"Clarke_{suffix}"

    _set_labels(plt, title_string)
    _set_axes_lengths(plt)
    _add_zone_lines(plt)
    _add_titles(plt)
    plt.savefig(filepath)

    plt.close()


def create_clarke_plots(
    zone, y_test, predictions, subject, predictors, window, model, horizon
):
    from .metric_clarke import _flatten_data

    _y_test, _predictions = _flatten_data(y_test, predictions, window, horizon)

    plot_filename = generate_file_name(subject, predictors, window, model=model)
    _plot_clarke_pie(zone, plot_filename, window, model)
    _plot_clarke_error(_y_test, _predictions, plot_filename)
