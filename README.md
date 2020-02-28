## Description
This repo is a rough set of functions that in the future, will constitute a Python package for calculating durations and onsets of stimuli from Presentation logs.

### Notebooks
* _Examples.ipynb_ - A Jupyter Notebook with examples of use

### Scripts
* _durons.py_ - A Python script with all the functions

### Files needed
* a Presentation-based .log file (I will upload anonymized files in the near future)

#### Important remarks

Our lab uses Presentation software, so at this point, the sole purpose of this set of functions is to transform the logs from this very software into SPM-ready onsets and durations of the stimuli presented during an fMRI experiment. Doing the same thing for PsychoPy-based experiments seems redundant at this point.

It is too early to build any package out of these functions as they have not been very heavily tested with logs from various procedures (which means the whole thing could break faster than Dexter's secret lab when Deedee pressed the red button).

#### Available functions:

| Function|Purpose| 
| :-------------:|:-------------:| 
| read_log|self-explanatory|
|fix_names|fix column names (remove spaces and other potentially destructive symbols|
|exclude_nonsubj_data|self-explanatory|
|stimuli_from_log|Show all unique stimuli/events from the file|
|pairwise_idx|Figure out where a single block starts and ends|
|list2regex|self-explanatory|
|get_blocks|Grab all the important events from the pre-defined blocks|
|calculate_times|The first fMRI pulse becomes the starting point, and all the events have their times recalculated and converted to seconds|
|durons|Calculate durations and onsets in pre-defined (or not) blocks|
|durons_savemat|Export durations and onsets to an SPM-ready .mat file|

