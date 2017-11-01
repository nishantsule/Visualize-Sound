# WaveCamNS
Acoustic wave solver coupled to webcam

This is a project was inspired from the work found here: https://github.com/audunsh/wavecam 
It demonstrates propagation and interaction of sound in different media with rigid boundaries. 
The boundaries are read from the webcam and the solution to the sound wave equation is superimposed on the frame that is read from the webcam.
The wave solver uses the finite-difference time-domain method to solve for pressure and velocity in two dimensions.
Besides being able to draw any arbitrary boundary that can be read from the webcam, the user is able to change source frequency, sound velocity, density of medium, temperature, and also record audio to use as the source.

The code has been written in Python 2.7 and uses the following packages:
OpenCV 3.3
PyAudio 0.2
Numpy 1.13
Matplotlib 2.0

Please send bug reports or feature requests to nsule@seas.harvard.edu
