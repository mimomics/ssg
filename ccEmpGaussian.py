"""Class-conditional empirical Gaussian model."""

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
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal as mvn

def ccEmpGaussianFit(Y, C):
  """Fit a class-conditional Gaussian model.

  Args:
    Y (array-like): Data matrix.
    C (array-like): Boolean vector of classes.

  Returns:
    dict: Parameters of the fitted model:
      pr0, pr1:     Class priors;
      mu0, mu1:     Class means;
      cova0, cova1: Class covariances.
  """
  i1 = np.asarray(C)
  i0 = ~i1

  Y0 = np.asarray(Y)[i0,:]
  Y1 = np.asarray(Y)[i1,:]

  pr0 = float(Y0.shape[0]) / Y.shape[0]
  pr1 = float(Y1.shape[0]) / Y.shape[0]

  mu0 = Y0.mean(axis=0)
  mu1 = Y1.mean(axis=0)

  cova0 = np.cov(Y0, rowvar=0)
  cova1 = np.cov(Y1, rowvar=0)

  return {
    'pr0': pr0,
    'pr1': pr1,
    'mu0': mu0,
    'mu1': mu1,
    'cova0': cova0,
    'cova1': cova1 }

def ccEmpGaussianPredict(model, Y):
  """Make a prediction using a trained ccEmpGaussian model.

  Args:
    model (tuple): The value returned from ccEmpGaussianFit.
    Y (array-like): The data matrix.

  Returns:
    array: Vector of probabilities for class 1.
  """
  lpr0 = np.log(model['pr0'])
  lpr1 = np.log(model['pr1'])

  llh0 = mvn.logpdf(Y, model['mu0'], model['cova0'])
  llh1 = mvn.logpdf(Y, model['mu1'], model['cova1'])

  fC = np.vstack([ llh0+lpr0, llh1+lpr1 ])
  lZ = logsumexp(fC, axis=0)
  return np.exp( fC[1,:] - lZ )
