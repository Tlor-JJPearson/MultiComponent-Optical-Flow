# MultiComponent-Optical-Flow
A Python script used in a thesis to calculate ans visualize optical flow for multiple chroma channels, using 4:2:0 YUV files.

It tests for seperate Y,U and V channels and also a channel made from averaging vectors from all three.

Language used is:
Python 3.11.4

This program has following modules installed (modules can be installed by pip):
- NumPy 1.26.3
- SciPy 1.12.0
- OpenCV_Python 4.9.0

Write command in Git Bash terminal inside the program's directory:
"python main.py" <-- has script running ericPrince's Farneback algorithm
optical_flow.py has this function 
GitHub of this specific solution: https://github.com/ericPrince/optical-flow 

"python new.py" <--  has script running Farneback algorithm from OpenCV


YUV_ folder has:
- YUV files (from 176x144 up to 352x288 - bigger could not fit)

Both "main.py" and "new.py" need to open files that actually exist -
please change the line "with open(.....) as f" (after a big comment) to an existing directory of a ".yuv" file.
