"""Implementation of Meinshausen and Buhlmann's stability selection method.

See the paper:
N. Meinshausen and P. Buhlmann (2010) Stability Selection
Journal of the Royal Statistical Society B, 72, Part 4, pp 417-473.
"""

# Copyright (c) [2017] Pharmatics Limited
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of Pharmatics Limited. The intellectual and technical concepts contained
# herein are proprietary to Pharmatics Limited and are protected by
# copyright law.
#
# Copying and exploitation of this software is permitted strictly for
# academic non-commercial research and teaching purposes only.
#
# Without limitation, any reproduction, modification, sub-licensing,
# redistribution of this software, or any commercial use of any part of
# this software including corporate or for-profit research or as the basis
# of commercial product or service provision, is forbidden unless
# a license is obtained from Pharmatics Limited by contacting
# info@pharmaticsltd.com.
#
# THE CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR PARTICULAR PURPOSE.
#
# IN NO EVENT SHALL PHARMATICS LIMITED BE LIABLE TO ANY PARTY FOR DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST
# PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS  DOCUMENTATION,
# EVEN IF PHARMATICS LIMITED HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#
# Author: Peter Orchard
# Date:   25-10-2017

import numpy as np

def _subsample(X):
  """Return a random subsample of X.

  Args:
    X: Data matrix or list of such.
      For each matrix Xn, the subsample size is floor( Xn.shape[0] / 2 ).

  Returns:
    Data matrix or list of such.
  """
  if isinstance(X, list):
    return [ _subsample(Xn) for Xn in X ]
  N  = X.shape[0]
  N2 = int( np.floor(N / 2) )
  i  = np.random.permutation(N)[0:N2]
  return X[i]

def stabilitySelection(X, T, fFit, fpr, fitPath=False, N=100):
  """Learn a model structure by stability selection.

  Args:
    X: Data matrix or list of such.
    T (array-like): Vector of values for the tuning parameter.
    fFit (function): Fitting function.
      Takes 2 parameters: a data matrix and tuning parameter.
      - If fitPath is False, the second argument is a single tuning parameter,
        and the function returns a boolean vector indicating the non-zero
        parameters in the fitted model.
      - If fitPath is True, the second argument is a vector of tuning
        parameters, and the function returns a matrix where each column
        indicates the non-zeros in the fitted model correponding to one
        tuning parameter.
    fpr (float): False positive rate.
      This will be used to set the threshold for stability selection.
    fitPath (bool): See fFit.
    N (int): Number of subsamples to use.

  Returns:
    array: Boolean vector indicating whether each variable is selected.
      Variables are ordered as in the return value of fFit.
  """
  T = np.asarray(T)

  # Each element will be a matrix where rows are model parameters and columns
  # are tuning parameters. Entries indicate whether a model parameter is
  # present (non-zero) in the model fitted with the associated tuning parameter.
  SList = []

  for n in range(N):
    Xn = _subsample(X)
    if fitPath:
      S = fFit(Xn, T)
    else:
      S = np.vstack([ fFit(Xn, t) for t in T ]).T
    SList.append(S)

  # Selection probabilities.
  P = np.mean( np.stack(SList, axis=2), axis=2 )

  # Set the threshold by approximately bounding the false positives.
  # See Section 2.4 of the paper.
  p = P.shape[0]
  q = P.mean(axis=0).max()
  threshold = ( (q*q)/(p*p*fpr) + 1 )/2

  # Stable variables.
  return P.max(axis=1) >= threshold
