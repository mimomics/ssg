"""Class-conditional graphical lasso model."""

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
from py2r import *
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from stability_selection import *

rLoadLibrary('glasso')

# =============================================================================
# Graphical Lasso
# =============================================================================

def _glassoFit(Y, pen, tol, maxIter, G=None):
  """Fit a graphical lasso model using the R glasso package.

  The graphical lasso paper is available here at the time of writing:
  http://statweb.stanford.edu/~tibs/ftp/graph.pdf
  Friedman, Jerome and Hastie, Trevor and Tibshirani, Robert (2008).
  "Sparse inverse covariance estimation with the graphical lasso"
  Biostatistics. Biometrika Trust.

  Args:
    Y (array-like): Data matrix.
    pen: L1 penalty. Scalar or matrix.
    tol (float): Tolerance (thr argument to glasso).
    maxIter (int): Maximum number of iterations (maxit argument to glasso).
    G (array): Boolean matrix describing a structure. Missing edges
      (False entries) will be forced to zero in the glasso solution.
  """
  rResult = genVarName()
  wi = rResult + '$wi'

  S = np.cov(Y, rowvar=False, bias=True)
  rS = pyArrayToRMatrix(S)

  rPen = genVarName()
  if np.isscalar(pen):
    r( '%s <- %f' % (rPen, pen) )
  else:
    pyArrayToRMatrix(pen, rPen)

  rZero = genVarName()
  if G is None:
    r( rZero + ' <- NULL' )
  else:
    # Form a 2-column matrix whose rows are the indices of the zero entries.
    # +1 because R indices are 1-based.
    Z = np.vstack( np.where(np.triu(~G, 1)) ).T + 1
    if len(Z) == 0:
      r( rZero + ' <- NULL' )
    else:
      pyArrayToRMatrix(Z, rZero)

  r( '%s <- glasso(s=%s, rho=%s, zero=%s, thr=%f, maxit=%d)' %
     (rResult, rS, rPen, rZero, tol, maxIter) )
  K = pyArrayFromRMatrix(wi)
  K = (K + K.T) / 2

  rRemove( [rResult, rS, rPen, rZero] )
  return K

# =============================================================================
# Class-Conditional Graphical Lasso
# =============================================================================

def ccGlassoFit(Y, C, penPrec0, penPrec1, tol=1e-4, maxIter=100):
  """Fit a class-conditional graphical lasso model.

  Args:
    Y (array-like): Data matrix.
    C (array-like): Boolean vector of classes.
    penPrec0 (float): Penalty on precision matrix of class-0 model.
    penPrec1 (float): Penalty on precision matrix of class-1 model.
    tol (float): Tolerance parameter passed to glasso.
    maxIter (int): Maximum iterations of the glasso algorithm.

  Returns:
    dict: Parameters of the fitted model:
      pr0, pr1: Priors.
      mu0, mu1: Means.
      prec0, prec1: Precision matrices.
  """
  # Ensure this function is not called for cross-validation.
  assert( np.isscalar(penPrec0) )
  assert( np.isscalar(penPrec1) )

  Y = np.asarray(Y)
  C = np.asarray(C).ravel().astype(bool)

  Y0 = Y[~C,:]
  Y1 = Y[ C,:]

  N  = Y.shape[0]
  N0 = Y0.shape[0]
  N1 = Y1.shape[0]

  model = {}

  model['pr0'] = N0 / N
  model['pr1'] = N1 / N

  model['mu0'] = Y0.mean(axis=0)
  model['mu1'] = Y1.mean(axis=0)

  model['prec0'] = _glassoFit(Y0, penPrec0, tol, maxIter)
  model['prec1'] = _glassoFit(Y1, penPrec1, tol, maxIter)

  return model

def ccGlassoPredict(model, Y):
  """Make a prediction using a trained ccGlasso model.

  Args:
    model (dict): The return value from ccGlassoFit.
    Y (array-like): Data matrix.

  Returns:
    array: Vector of probabilities for class 1.
  """
  Y = np.asarray(Y)

  lpr0 = np.log( model['pr0'] )
  lpr1 = np.log( model['pr1'] )
  mu0 = model['mu0']
  mu1 = model['mu1']
  cova0 = np.linalg.inv( model['prec0'] )
  cova1 = np.linalg.inv( model['prec1'] )

  llh0 = mvn.logpdf(Y, mu0, cova0)
  llh1 = mvn.logpdf(Y, mu1, cova1)

  fC = np.vstack([ llh0+lpr0, llh1+lpr1 ])
  lZ = logsumexp(fC, axis=0)
  return np.exp( fC[1,:] - lZ )

# =============================================================================
# Cross-Validation of the Penalties
# =============================================================================

def ccGlassoCV(
  Y, C,
  penRange,
  nFolds,
  standardise=False,
  tol=1e-4,
  maxIter=100 ):
  """Fit a ccGlasso model using the best parameters found by CV of the AUC.

  Args:
    Y (array-like): Data matrix.
    C (array-like): Boolean vector of classes.
    penRange (array-like): 2-column matrix, where each row is a pair
      (penPrec0, penPrec1) of penalties to test under cross-validation.
    nFolds (int): Number of CV folds.
    standardise (bool): If True, transform the training data to zero mean
      and unit variance, for each fold. Apply the same transform to test data.
    tol (float): Tolerance parameter passed to GraphLasso.
    maxIter (int): Maximum iterations of the GraphLasso algorithm.

  Returns:
    dict: Fitted model using the best chosen parameters.
    float: Best penalty found by CV for class 0.
    float: Best penalty found by CV for class 1.
  """
  Y = np.asarray(Y)
  C = np.asarray(C).ravel().astype(bool)

  # Convert 2-column matrix into a list of tuples.
  penRange = [ tuple(row) for row in np.asarray(penRange) ]

  folds = StratifiedKFold(nFolds, shuffle=True)
  aucSum = np.zeros( len(penRange) )
  for iTrain, iTest in folds.split(Y, C):
    YTrain = Y[iTrain,:]
    CTrain = C[iTrain]
    YTest = Y[iTest,:]
    CTest = C[iTest]

    if standardise:
      ss = StandardScaler().fit(YTrain)
      YTrain = ss.transform(YTrain)
      YTest  = ss.transform(YTest)

    for iPen in range( len(penRange) ):
      pen0, pen1 = penRange[iPen]
      model = ccGlassoFit(YTrain, CTrain, pen0, pen1, tol, maxIter)
      predictions = ccGlassoPredict(model, YTest)
      aucSum[iPen] += roc_auc_score(CTest, predictions)

  iBest = np.argmax(aucSum)
  penBest0, penBest1 = penRange[iBest]
  modelBest = ccGlassoFit(Y, C, penBest0, penBest1, tol, maxIter)

  return modelBest, penBest0, penBest1

# =============================================================================
# Learning Initial Structure by Thresholding Partial Correlations
# =============================================================================

def _matToVec(m):
  """Pull out the upper triangle of a matrix, as a vector.

  Does not include the main diagonal.
  """
  return m[ np.triu_indices_from(m, 1) ]

def _vecToMat(v, diagFill=True):
  """Reverse operation of _matToVec.

  Args:
    v (array): Vector to convert.
    diagFill: Value with which to fill the main diagonal.
      (Necessary because _matToVec loses information about the diagonal.)

  Returns:
    array: Reconstructed matrix.
  """
  p = int( (1 + np.sqrt(1+8*len(v))) / 2 )
  m = np.zeros( [p, p], dtype=bool )
  m[ np.triu_indices_from(m, 1)  ] = v
  m = m + m.T
  np.fill_diagonal(m, diagFill)
  return m

def _pcorrThreshSS(Y, threshold):
  """Return a structure by thresholding the empirical partial correlations."""
  # Compute the partial correlation matrix.
  # If S is not invertible (because it is not positive-definite),
  # boost its diagonal until it is.
  S = np.cov(Y, rowvar=False)
  while True:
    try:
      K = np.linalg.inv(S)
      break
    except np.linalg.LinAlgError:
      S = S + np.eye( S.shape[0] )
  d = np.sqrt( np.diag(K) )
  P = -((K / d).T / d).T
  P = (P + P.T) / 2
  np.fill_diagonal(P, 1)

  G = np.abs(P) >= threshold
  return _matToVec(G)

def ccGlassoPreSS(
  Y, C,
  thresholdRange, fpr,
  penPrec0, penPrec1, tol=1e-4, maxIter=100 ):
  """Use thresholding with stability selection to find structures, then glasso.

  The stability selection paper is available here at the time of writing:
  http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00740.x/full
  Meinshausen, Nicolai, and Peter Bühlmann.
  "Stability selection."
  Journal of the Royal Statistical Society: Series B (Statistical Methodology)
  72.4 (2010): 417-473.

  For each class, compute the empirical partial correlation matrix, then
  threshold the absolute values to find the structure. Do this with stability
  selection to improve the results, and to avoid the need to choose a single
  threshold. Having found the structure, fix the zeros in place and run glasso
  on each class to fit the precision matrices.

  Args:
    Y (array-like): Data matrix.
    C (array-like): Boolean vector of classes.
    thresholdRange (array-like): Vector of thresholds.
      This is the tuning parameter for stability selection.
    fpr (float): False positive rate parameter to stability selection.
    penPrec0 (float): Penalty on precision matrix of class-0 model.
    penPrec1 (float): Penalty on precision matrix of class-1 model.
    tol (float): Tolerance parameter passed to glasso.
    maxIter (int): Maximum iterations of the glasso algorithm.

  Returns:
    dict: Parameters of the fitted model:
      pr0, pr1: Priors.
      mu0, mu1: Means.
      prec0, prec1: Precision matrices.
  """
  Y = np.asarray(Y)
  C = np.asarray(C).ravel().astype(bool)

  Y0 = Y[~C,:]
  Y1 = Y[ C,:]

  G0 = _vecToMat( stabilitySelection(Y0, thresholdRange, _pcorrThreshSS, fpr) )
  G1 = _vecToMat( stabilitySelection(Y1, thresholdRange, _pcorrThreshSS, fpr) )

  N  = Y.shape[0]
  N0 = Y0.shape[0]
  N1 = Y1.shape[0]

  model = {}

  model['pr0'] = N0 / N
  model['pr1'] = N1 / N

  model['mu0'] = Y0.mean(axis=0)
  model['mu1'] = Y1.mean(axis=0)

  model['prec0'] = _glassoFit(Y0, penPrec0, tol, maxIter, G=G0)
  model['prec1'] = _glassoFit(Y1, penPrec1, tol, maxIter, G=G1)

  return model
