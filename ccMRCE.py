"""Class-conditional MRCE model."""

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
from stability_selection import *

rLoadLibrary('MRCE', quiet=True)

# =============================================================================
# MRCE Wrapper
# =============================================================================

def mrceFit(X, Y, penMu, penPrec, nFolds=None):
  """Fit an MRCE model.

  The MRCE paper is available here at the time of writing:
  https://pdfs.semanticscholar.org/2e0e/62a43e55f84c4191ad6dccd7d51748ec6f80.pdf
  Rothman, Adam J., Elizaveta Levina, and Ji Zhu.
  "Sparse multivariate regression with covariance estimation."
  Journal of Computational and Graphical Statistics 19.4 (2010): 947-962.

  The penalties may be scalars or lists of scalars.
  In the latter case, the best parameters will be chosen by CV.

  Args:
    X (array-like): Predictors.
    Y (array-like): Targets.
    penMu: Penalty on mean parameters.
    penPrec: Penalty on precision parameters.
    nFolds (int): Number of folds if using cross-validation.

  Returns:
    R variable name of the fitted model.
  """
  rModel = genVarName()
  rX = pyArrayToRMatrix(X)
  rY = pyArrayToRMatrix(Y)

  if np.isscalar(penMu) and np.isscalar(penPrec):
    r( '%s <- mrce(%s, %s, lam1=%s, lam2=%s, method="single")' %
      (rModel, rX, rY, repr(penPrec), repr(penMu)) )

  else:
    assert(nFolds is not None)
    rPenMu = pyArrayToRVector(penMu)
    rPenPrec = pyArrayToRVector(penPrec)
    r( '%s <- mrce(%s, %s, lam1.vec=%s, lam2.vec=%s, method="cv", kfold=%s)' %
      (rModel, rX, rY, rPenPrec, rPenMu, repr(nFolds)) )
    rRemove( [rPenMu, rPenPrec] )

  rRemove( [rX, rY] )
  return rModel

# =============================================================================
# Class-Conditional MRCE
# =============================================================================

def ccmrceFit(X, Y, C, penMu0, penMu1, penPrec0, penPrec1):
  """Fit a class-conditional MRCE model.

  Args:
    X (array-like): Predictors.
    Y (array-like): Targets.
    C (array-like): Boolean vector of classes.
    penMu0: Penalty on mean parameters of class-0 model.
    penMu1: Penalty on mean parameters of class-1 model.
    penPrec0: Penalty on precision parameters of class-0 model.
    penPrec1: Penalty on precision parameters of class-1 model.

  Returns:
    tuple: Tuple with 4 elements:
      R variable names for the class-0 and class-1 models,
      followed by the priors for class 0 and class 1.
  """
  # Ensure this function is not called for cross-validation.
  # Use ccmrceCV for that.
  assert( np.isscalar(penMu0) )
  assert( np.isscalar(penMu1) )
  assert( np.isscalar(penPrec0) )
  assert( np.isscalar(penPrec1) )

  i1 = np.asarray(C)
  i0 = ~i1

  X0 = np.asarray(X)[i0,:]
  Y0 = np.asarray(Y)[i0,:]
  X1 = np.asarray(X)[i1,:]
  Y1 = np.asarray(Y)[i1,:]

  pr0 = float(Y0.shape[0]) / Y.shape[0]
  pr1 = float(Y1.shape[0]) / Y.shape[0]

  rM0 = mrceFit(X0, Y0, penMu0, penPrec0)
  rM1 = mrceFit(X1, Y1, penMu1, penPrec1)

  return rM0, rM1, pr0, pr1

def mrceLogLikelihood(rM, X, Y):
  """Compute the log likelihood of the data under a trained MRCE model."""
  mu = pyArrayFromRVector(rM + '$muhat')
  B  = pyArrayFromRVector(rM + '$Bhat')

  # The mrce trainer appears to include a "sigma" field in the returned
  # list when "method" == "single", but not for "cv". So we do not
  # use that, but invert the "omega" field, which is always present.
  S = np.linalg.inv( pyArrayFromRMatrix(rM + '$omega') )

  # mvn.logpdf cannot take a different mean for each data point.
  # So subtract the means, and pass zero for the mean parameter.
  return mvn.logpdf( Y - (mu + np.dot(X, B)), np.zeros_like(mu), S )

def ccmrcePredict(rM0, rM1, pr0, pr1, X, Y):
  """Make predictions using a trained ccMRCE model.

  Args:
    rM0 (String): R variable name of the class-0 model.
    rM1 (String): R variable name of the class-1 model.
    pr0 (float): Prior probability of class 0.
    pr1 (float): Prior probability of class 1.
    X (array-like): Inputs.
    Y (array-like): Outputs.

  Returns:
    array: Probability of each example belonging to class 1.
  """
  lpr0 = np.log(pr0)
  lpr1 = np.log(pr1)

  llh0 = mrceLogLikelihood(rM0, X, Y)
  llh1 = mrceLogLikelihood(rM1, X, Y)

  fC = np.vstack([ llh0+lpr0, llh1+lpr1 ])
  lZ = logsumexp(fC, axis=0)
  return np.exp( fC[1,:] - lZ )

# =============================================================================
# Cross-Validation of the Penalties
# =============================================================================

def ccmrceCV(X, Y, C, penRange, nFolds):
  """Fit a ccMRCE model using the best parameters found by CV of the AUC.

  Args:
    X (array-like): Predictors.
    Y (array-like): Targets.
    C (array-like): Boolean vector of classes.
    penRange (array-like): 4-column matrix, where each row is a tuple
      (penMu0, penMu1, penPrec0, penPrec1) of penalties to test under
      cross-validation.
    nFolds (int): Number of CV folds.

  Returns:
    tuple: Fitted model, as returned by ccmrceFit,
      using the best penalties chosen by CV.
    tuple: Best penalties chosen by CV.
  """
  X = np.asarray(X)
  Y = np.asarray(Y)
  C = np.asarray(C).ravel().astype(bool)

  # Convert 4-column matrix into a list of tuples.
  penRange = [ tuple(row) for row in np.asarray(penRange) ]

  folds = StratifiedKFold(nFolds, shuffle=True)
  aucSum = np.zeros( len(penRange) )
  for iTrain, iTest in folds.split(Y, C):
    XTrain = X[iTrain,:]
    YTrain = Y[iTrain,:]
    CTrain = C[iTrain]
    XTest = X[iTest,:]
    YTest = Y[iTest,:]
    CTest = C[iTest]

    for iPen in range( len(penRange) ):
      pen = penRange[iPen]
      model = ccmrceFit(XTrain, YTrain, CTrain, *pen)
      predictions = ccmrcePredict(*model, XTest, YTest)
      aucSum[iPen] += roc_auc_score(CTest, predictions)
      rRemove([ model[0], model[1] ])

  iBest = np.argmax(aucSum)
  penBest = penRange[iBest]
  modelBest = ccmrceFit(X, Y, C, *penBest)

  return modelBest, penBest
