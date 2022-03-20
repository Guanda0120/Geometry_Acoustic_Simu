# Geometry_Acoustic_Simu

## AdaptiveDSP
### Least Square Method
#### Ref of this Method

This methed is from Room Acoustic Edition 6

> Usually, the level L(t) in the experimental decay curves does not fall in a strictly linear way but contains random uctuations which are due, as explained in Section 3.8, to complicated interferences between decaying normal modes. If these  uctuations are not too strong, the decay curve can be approximated by a straight line. This can be done manually, that is, by ruler and pencil. If high precision is required it may be advantageous to carry out a ‘least square  t’: Let t1 and t2 denote the interval in which the decay curve is to be approximated (see Figure 8.11). Then, the following integrations must be performed:

#### Main interpretation of this algrithm
This method basicly about linear regression

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\omega=(X^{T}X)^{-1}X^{T}y" title="\Large \omega=(X^{T}X)^{-1}X^{T}y" />

where X is input signal: time(s) combined with Biased Term horizontally

<img src="https://latex.codecogs.com/svg.image?X=\begin{bmatrix}&space;x_{0}&space;&&space;b&space;\\\vdots&space;&&space;\vdots\\&space;x_{n-1}&space;&&space;b&space;\end{bmatrix}" title="X=\begin{bmatrix}&space;x_{0}&space;&&space;b&space;\\\vdots&space;&&space;\vdots\\&space;x_{n-1}&space;&&space;b&space;\end{bmatrix}" />

where x[0]~x[n-1] is a n length discrete time, b is the biased term b=1
and y is the discrete signal Sound Preasure Level(dB)

#### Dis of this algrithm
