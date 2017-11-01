"""Class-conditional Gaussian Graphical model.

This module utilises the R package "longitudinal", which implements the method
described in the following paper:
R. Opgen-Rhein and K. Strimmer. 2006. Using regularized dynamic correlation
to infer gene dependency networks from time-series microarray data.
Proceedings of the 4th International Workshop on Computational Systems Biology,
WCSB 2006 (June 12-13, 2006, Tampere, Finland), pp. 73-76.
It is available here at time of writing:
http://strimmerlab.org/publications/conferences/dyncorshrink2006.pdf

The ccGGM is based on the above method, and was developed by Elisa Benedetti
and Jan Krumsiek for differential network analysis.

Peter Orchard converted the code to Python, and slightly generalised the
method to allow predictions to be made with the learned structures.
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

from py2r import *
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal as mvn

rLoadLibrary('longitudinal')

def ccggmFit(X, Y, C):
  """Fit the class-conditional joint Gaussian model.

  Args:
    X (array-like): Predictors, may be None.
    Y (array-like): Targets.
    C (array-like): Boolean vector of classes.

  Returns:
    dict: Has the following fields:
      pX - Dimensionality of the predictors.
      pY - Dimensionality of the targets.
      pr0, pr1 - Class priors.
      mu0, mu1 - Gaussian means.
      cova0, cova1 - Gaussian covariances.
      The model is fitted to the joint of targets and predictors;
      the predictors come first in mu* and cova*.
  """
  pX = X.shape[1]
  pY = Y.shape[1]

  if X is None:
    XY = np.asarray(Y)
  else:
    XY = np.hstack( [np.asarray(X), np.asarray(Y)] )

  i1 = np.asarray(C).astype(bool)
  i0 = ~i1

  # Use the empirical class ratio for the class prior.
  pr0 = float( i0.sum() ) / C.shape[0]
  pr1 = 1 - pr0

  XY0 = XY[i0,:]
  XY1 = XY[i1,:]

  mu0 = XY0.mean(axis=0)
  mu1 = XY1.mean(axis=0)

  rCova0, rCova1 = genVarName(2)
  rXY0 = pyArrayToRMatrix(XY0)
  rXY1 = pyArrayToRMatrix(XY1)

  r( '%s = dyn.cov(%s)' % (rCova0, rXY0) )
  r( '%s = dyn.cov(%s)' % (rCova1, rXY1) )

  cova0 = pyArrayFromRMatrix(rCova0)
  cova1 = pyArrayFromRMatrix(rCova1)

  rRemove( [rCova0, rCova1, rXY0, rXY1] )

  return { 'pX': pX, 'pY': pY,
           'pr0': pr0, 'mu0': mu0, 'cova0': cova0,
           'pr1': pr1, 'mu1': mu1, 'cova1': cova1 }

def ccggmPredict(ggm, X, Y):
  """Make predictions using the trained class-conditional GGM model.
  
  Args:
    ggm (dict): Return value from ccggmFit.
    X (array-like): Predictors, may be None.
    Y (array-like): Targets.

  Returns:
    array: Vector of probabilities for class 1.
  """
  if X is None:
    XY = np.asarray(Y)
  else:
    XY = np.hstack( [np.asarray(X), np.asarray(Y)] )

  lpr0 = np.log( ggm['pr0'] )
  lpr1 = np.log( ggm['pr1'] )

  llh0 = mvn.logpdf( XY, ggm['mu0'], ggm['cova0'] )
  llh1 = mvn.logpdf( XY, ggm['mu1'], ggm['cova1'] )

  fC = np.vstack([ llh0+lpr0, llh1+lpr1 ])
  lZ = logsumexp(fC, axis=0)
  return np.exp( fC[1,:] - lZ )
