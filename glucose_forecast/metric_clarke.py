"""
CLARKE ERROR GRID ANALYSIS      ClarkeErrorGrid.py

Need Matplotlib Pyplot


The Clarke Error Grid shows the differences between a blood glucose predictive measurement and a reference measurement,
and it shows the clinical significance of the differences between these values.
The x-axis corresponds to the reference value and the y-axis corresponds to the prediction.
The diagonal line shows the prediction value is the exact same as the reference value.
This grid is split into five zones. Zone A is defined as clinical accuracy while
zones C, D, and E are considered clinical error.

Zone A: Clinically Accurate
    This zone holds the values that differ from the reference values no more than 20 percent
    or the values in the hypoglycemic range (<70 mg/dl).
    According to the literature, values in zone A are considered clinically accurate.
    These values would lead to clinically correct treatment decisions.

Zone B: Clinically Acceptable
    This zone holds values that differe more than 20 percent but would lead to
    benign or no treatment based on assumptions.

Zone C: Overcorrecting
    This zone leads to overcorrecting acceptable BG levels.

Zone D: Failure to Detect
    This zone leads to failure to detect and treat errors in BG levels.
    The actual BG levels are outside of the acceptable levels while the predictions
    lie within the acceptable range

Zone E: Erroneous treatment
    This zone leads to erroneous treatment because prediction values are opposite to
    actual BG levels, and treatment would be opposite to what is recommended.


SYNTAX:
        plot, zone = clarke_error_grid(ref_values, pred_values, title_string)

INPUT:
        ref_values          List of n reference values.
        pred_values         List of n prediciton values.
        title_string        String of the title.

OUTPUT:
        plot                The Clarke Error Grid Plot returned by the function.
                            Use this with plot.show()
        zone                List of values in each zone.
                            0=A, 1=B, 2=C, 3=D, 4=E

EXAMPLE:
        plot, zone = clarke_error_grid(ref_values, pred_values, "00897741 Linear Regression")
        plot.show()

References:
[1]     Clarke, WL. (2005). "The Original Clarke Error Grid Analysis (EGA)."
        Diabetes Technology and Therapeutics 7(5), pp. 776-779.
[2]     Maran, A. et al. (2002). "Continuous Subcutaneous Glucose Monitoring in Diabetic
        Patients" Diabetes Care, 25(2).
[3]     Kovatchev, B.P. et al. (2004). "Evaluating the Accuracy of Continuous Glucose-
        Monitoring Sensors" Diabetes Care, 27(8).
[4]     Guevara, E. and Gonzalez, F. J. (2008). Prediction of Glucose Concentration by
        Impedance Phase Measurements, in MEDICAL PHYSICS: Tenth Mexican
        Symposium on Medical Physics, Mexico City, Mexico, vol. 1032, pp.
        259261.
[5]     Guevara, E. and Gonzalez, F. J. (2010). Joint optical-electrical technique for
        noninvasive glucose monitoring, REVISTA MEXICANA DE FISICA, vol. 56,
        no. 5, pp. 430434.


Added data handling function
Andras Novoszath
April 20 2020

Originally made by:
Trevor Tsue
7/18/17

Based on the Matlab Clarke Error Grid Analysis File Version 1.2 by:
Edgar Guevara Codina
codina@REMOVETHIScactus.iico.uaslp.mx
March 29 2013
"""


import os


def _validate_length(ref_values, pred_values):
    # Checking to see if the lengths of the reference and prediction arrays are the same
    assert len(ref_values) == len(
        pred_values
    ), "Unequal number of values (reference : {}) (prediction : {}).".format(
        len(ref_values), len(pred_values)
    )


def _validate_value_range(ref_values, pred_values):
    # Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400 or max(pred_values) > 400:
        print(
            "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                max(ref_values), max(pred_values)
            )
        )
    if min(ref_values) < 0 or min(pred_values) < 0:
        print(
            "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(
                min(ref_values), min(pred_values)
            )
        )


def _calc_clarke_zones(ref_values, pred_values):
    _validate_length(ref_values, pred_values)
    _validate_value_range(ref_values, pred_values)

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (
            pred_values[i] <= 1.2 * ref_values[i]
            and pred_values[i] >= 0.8 * ref_values[i]
        ):
            zone[0] += 1  # Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (
            ref_values[i] <= 70 and pred_values[i] >= 180
        ):
            zone[4] += 1  # Zone E

        elif (
            (ref_values[i] >= 70 and ref_values[i] <= 290)
            and pred_values[i] >= ref_values[i] + 110
        ) or (
            (ref_values[i] >= 130 and ref_values[i] <= 180)
            and (pred_values[i] <= (7 / 5) * ref_values[i] - 182)
        ):
            zone[2] += 1  # Zone C
        elif (
            (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180))
            or (
                ref_values[i] <= 175 / 3
                and pred_values[i] <= 180
                and pred_values[i] >= 70
            )
            or (
                (ref_values[i] >= 175 / 3 and ref_values[i] <= 70)
                and pred_values[i] >= (6 / 5) * ref_values[i]
            )
        ):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    return zone


def _flatten_data(y_test, predictions, window_size, horizon):
    """Creates a list of lists(?) from the rows of the predictions and test arrays.
    """
    i = window_size
    _predictions = list(predictions[0])
    _y_test = list(y_test[0])
    i = i + horizon
    while i < y_test.shape[0] - horizon:
        _predictions = _predictions + list(predictions[i])
        _y_test = _y_test + list(y_test[i])
        i = i + horizon

    return _y_test, _predictions


def generate_clarke_errors(y_test, predictions, window_size, horizon):
    # Calculates and plot clarke error for predictions data > horizon

    _y_test, _predictions = _flatten_data(y_test, predictions, window_size, horizon)

    if not os.path.exists("Figures"):
        os.makedirs("Figures")


    zone = _calc_clarke_zones(_y_test, _predictions)

    return ((zone[0] + zone[1]) / sum(zone)) * 100, zone
