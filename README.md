for stat_arb_simulator_vertex.py

25% BTC, 75% trading script results in:

"""
 message: Optimization terminated successfully
 success: True
  status: 0
     fun: -0.07339428821319546
       x: [ 2.273e-01  7.727e-01]
     nit: 12
     jac: [-3.434e-03  1.010e-03]
    nfev: 46
    njev: 12
daily return w/ futures leverage (20x): 0.0068740981056503135 (0.68%)
daily stddev w/ futures leverage (20x): 0.0936598511001627 (9%)

Optimal Portfolio Sortino Ratio: 0.0734
[ 4.98290554e-04  3.19387855e-03  1.18855526e-02  2.05691679e-02
 -1.40126775e-02  3.70617190e-03 -2.49586378e-03 -1.27723742e-02
  2.95470601e-03 -8.02763009e-03  1.58840614e-03  9.60022961e-03
 -7.69861201e-03  3.08749583e-02  9.45271419e-03 -6.13164246e-03
  7.19777461e-04 -1.02174604e-02  8.66375540e-03  1.78411251e-02
 -1.25170063e-02  2.59106838e-04 -2.85512410e-03  3.54566612e-03
  3.10641311e-03  1.47116660e-03  9.76242223e-03  1.11101196e-02
 -1.75337245e-03 -2.41140430e-03 -4.95635774e-03 -1.00078725e-03
  5.99128155e-03 -5.77806549e-04 -1.07142178e-02  4.72061983e-06
  4.65779364e-04 -3.90254124e-03 -1.20346097e-02 -7.42171866e-03
  8.58783463e-03 -2.84725460e-03  2.86377587e-03 -1.78954398e-04
 -1.95586569e-03  7.37182160e-03 -1.40955268e-02 -1.85673923e-03
 -6.12176541e-03  2.10796404e-03 -1.29317372e-03 -1.87876602e-04
  6.75704773e-05  4.85241099e-03  9.34986702e-03 -4.24155697e-03
 -8.33557528e-03 -1.35329163e-03 -2.53471290e-03 -2.17346507e-02
  1.25263286e-02 -7.00739193e-03  6.73513933e-04  9.95885956e-03
 -3.53960287e-03 -3.72501735e-03  1.54016891e-02 -2.90624768e-03
  1.82692944e-02 -1.66110670e-03 -7.43323149e-03 -2.93707062e-03
 -6.70061584e-03  1.86585109e-02 -1.31907346e-02 -2.91815495e-03
 -3.24287518e-04 -1.51115409e-04 -2.98574012e-03  4.15635237e-05
  2.20837171e-03 -3.94311259e-03  5.07708671e-03 -2.93078147e-03
  2.08235812e-03  2.68043883e-03 -3.27385676e-03 -9.49359005e-04
 -5.55025784e-04  2.60794766e-03  3.99513207e-03]
 """

Hourly predictions, forward tested on past 5mo.

Win rate ~ 56% over ~300 samples.
