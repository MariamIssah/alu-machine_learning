# Plotting with Matplotlib 📊

This project introduces basic data visualization techniques using Python's `matplotlib` library. You'll learn how to create different types of plots such as line graphs, scatter plots, histograms, and bar charts.

## Project Structure

Each script is located in the `math/plotting` directory and follows the naming convention: `0-line.py`, `1-scatter.py`, etc.

---

📈 Tasks Overview

0.  Line Graph

- File: `0-line.py`
- Description: Plots the cube of numbers from 0 to 10 as a solid red line.
- Techniques Used: `plt.plot`, axis limit setup.

1. Scatter Plot

- File: `1-scatter.py`
- Description: Plots a scatter plot representing height vs weight of men.
- Techniques Used: `plt.scatter`, axis labeling, `plt.title`.

2. Change of Scale

- File: `2-change_scale.py`
- Description: Shows the exponential decay of Carbon-14 with a logarithmic y-axis.
- Techniques Used: `plt.yscale('log')`, exponential decay formula.

3. Two is Better Than One

- File: `3-two.py`
- Description: Compares the decay of Carbon-14 and Radium-226.
- Techniques Used: Multiple line plots, `plt.legend`.

4. Frequency Histogram

- File: `4-frequency.py`
- Description: Displays the frequency of grades for a project using a histogram.
- Techniques Used: `plt.hist`, bin setup, `edgecolor`.

5. All in One

- File: `5-all_in_one.py`
- Description: Combines all previous plots into one figure using subplots.
  Techniques Used: `plt.subplot2grid`, `plt.suptitle`, `fontsize='x-small'`.

6. Stacked Bar Chart

- File: `6-bars.py`
- Description: Stacked bar chart showing different fruits held by three people.
- Techniques Used: `plt.bar`, stacking, color customization, legend.

---

🛠 Requirements

- Python 3.5+
- Numpy 1.15
- Matplotlib 3.0

Install the required packages:

```bash
pip install --user matplotlib==3.0 numpy==1.15 Pillow
sudo apt-get install python3-tk
```
