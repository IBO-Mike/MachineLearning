# TCAV Auditing Project

This project explores the use of TCAV as an auditing signal for model behavior.
The goal is to study how stable TCAV results are under different design choices,
including concept sets, control sets, and network layers.

## Folder Structure

- data/  
  Contains all raw data used in the experiments, including target class images,
  concept examples, and control examples. No code or results are stored here.

- experiments/  
  Contains all notebooks and scripts used to run TCAV experiments.
  This folder records how the experiments are conducted.

- results/  
  Stores outputs of the experiments, such as TCAV scores, tables, and figures.
  Only processed results are placed here.

- reflection.md  
  Includes reading summaries of related papers and a written reflection
  on the TCAV experiments and their implications.

- schedule.txt  
  A simple plan outlining the timeline of the project.

## Workflow

1. Prepare data in the `data/` folder.
2. Run TCAV experiments using scripts in `experiments/`.
3. Save experiment outputs in the `results/` folder.
4. Analyze results and write reflections in `reflection.md`.

This structure is intended to keep data, code, and analysis clearly separated.