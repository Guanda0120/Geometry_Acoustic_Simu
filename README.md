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
**1. It may not always linear decay. Just like graph below**
![alt text](https://github.com/Guanda0120/Geometry_Acoustic_Simu/blob/main/MdImg/NonLinear_Decay.png?raw=true)

**2. It may fluctuation strongly**
![alt text](https://github.com/Guanda0120/Geometry_Acoustic_Simu/blob/main/MdImg/StrongFluc.png?raw=true)

### Back Integral Method
#### Ref of this Method
This methed is from New Method of Measuring Reverberation Time M.R. Schroeder Bell Telephone Laboratories

#### Main interpretation of this algrithm

1. The main idea of Back Integral is **Convolution Theory**.Given a signal in time domain f(x) and IR g(x) 
<img src="https://latex.codecogs.com/svg.image?h(x)=\int_{-\infty}^{&plus;\infty}f(\tau)g(x-\tau)d\tau" title="\Large https://latex.codecogs.com/svg.image?h(x)=\int_{-\infty}^{&plus;\infty}f(\tau)g(x-\tau)d\tau" />

>If random noise is used as an excitation signal, each member of a series of repeated decay measurements is slightly different from all others and none of them is representative for all decay processes. This is an immediate consequence of the random character of the input signal. This uncertainty can be avoided, in principle, by averaging over a great number of individual reverberation curves, taken under otherwise unchanged conditions. Fortunately, this tedious procedure can be circumvented by applying an elegant method, called ‘backward integration’, which was proposed and first applied by M. R. Schroeder.13

In decreste time, it can express as:

<img src="https://latex.codecogs.com/svg.image?h[x]=\sum_{k=0}^{k=n-1}f[k]g[x-k]" title="\Large https://latex.codecogs.com/svg.image?h[x]=\sum_{k=0}^{k=n-1}f[k]g[x-k]" />

2. Change Sound Preasure Level to Preasure
3. Creat a IR:g(x). In ideal test process, when switch of the loudspeaker in time t_0, the sound preasure this momet is p_0 is inital sound preasure, and sound preasure=0 after t>t_0. So it is p_0 from 1~t_0, and 0 Pa from t_0+1 to end. And t_0 = t_end-t_0. It can express as:

<img src="https://latex.codecogs.com/svg.image?(1-u[0])\times&space;p_{init}" title="\Large https://latex.codecogs.com/svg.image?(1-u[0])\times&space;p_{init}" />
4. Convolve Processing:

<img src="https://latex.codecogs.com/svg.image?h[n]=\sum_{k=n}^{N-1}f[k]g[n-k]" title="\Large https://latex.codecogs.com/svg.image?h[n]=\sum_{k=n}^{N-1}f[k]g[n-k]" />
5. Vectorization:
h[n] = X·G. Where X is the sound preasure. and G is the triangle matrix times init preasure:
<img src="https://latex.codecogs.com/svg.image?h[n]=\begin{bmatrix}x[0]\\x[1]\\\vdots\\x[N-1]\end{bmatrix}^{T}\cdot\begin{bmatrix}&space;1&&space;0&space;&&space;\ldots&space;&0&space;&space;\\&space;1&&space;1&space;&&space;&space;&&space;&space;\vdots\\&space;\vdots&&space;&space;&1&space;&space;&0&space;&space;\\1&space;&&space;\ldots&space;&1&space;&1&space;&space;\\\end{bmatrix}&space;\times&space;p_{init}" title="\Large https://latex.codecogs.com/svg.image?h[n]=\begin{bmatrix}x[0]\\x[1]\\\vdots\\x[N-1]\end{bmatrix}^{T}\cdot\begin{bmatrix}&space;1&&space;0&space;&&space;\ldots&space;&0&space;&space;\\&space;1&&space;1&space;&&space;&space;&&space;&space;\vdots\\&space;\vdots&&space;&space;&1&space;&space;&0&space;&space;\\1&space;&&space;\ldots&space;&1&space;&1&space;&space;\\\end{bmatrix}&space;\times&space;p_{init}" />


### AdaptiveDSP Result
Use LSQ method 1k Hz Reverb Time is  0.39 s

Use Back Integral method 1k Hz Reverb Time(T20) is  0.35 s

Use Back Integral method 1k Hz Reverb Time(T30) is  0.33 s

Use Back Integral method 1k Hz Reverb Time(T60) is  0.43 s

Use Eyling Equation, 1k Hz Reverb Time is 0.49s
![alt text](https://github.com/Guanda0120/Geometry_Acoustic_Simu/blob/main/MdImg/1kHz_Decay_Curve_LSQ.png?raw=true)
