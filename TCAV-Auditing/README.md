# TCAV Auditing Project

This project studies TCAV as an auditing signal for model behavior.

The project consists of three parts: data preparation, TCAV experiments, and result analysis with reflections.

---

## Folder Structure

### data/

This folder contains all raw data used in the experiments.

- concepts/  
  Concept image sets used to construct TCAV concepts (e.g., striped).

- control_pool/  
  Image pool for generating random control sets in TCAV experiments.

- target/  
  Target class images.

---

### experiments/

This folder contains all scripts and notebooks used to run TCAV experiments.

- cav_exp1/  
  Files related to Experiment

- **TCAV_Auditing_Results.ipynb** 

  Main notebook for aggregating results, computing statistics, and producing tables.

- tcav_utils.py  
  Utility functions for running TCAV, loading data, and organizing results.

- tcav_plot.py  
  Functions for visualizing TCAV scores.

---

### reflection/

This folder contains written analysis and reflections.

- experiment-reflection.pdf  
  Written analysis and discussion of the TCAV experimental results.

- paper-reading-summary.pdf  
  Reading summaries of papers.

- reflection-table.csv  
  TCAV results scores.
