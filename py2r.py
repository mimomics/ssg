"""Functions for moving data between Python and R."""

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
import uuid

# Importing readline is a workaround for a problem that can occur when trying
# to import rpy2.robjects from a python script run from the Linux shell.
# This may fail with an undefined symbol 'PC'. The problem seems to be that the
# system readline library links to libtinfo, while the conda readline does not.
import readline
import rpy2.robjects as ro
r = ro.r

def rLoadLibrary(libName, quiet=False):
  """Load an R library into the R workspace.

  Args:
    libName (str): Name of the R library.
    quiet (bool): If True, suppress library start-up messages.
  """
  if quiet:
    r( 'suppressPackageStartupMessages(library(' + libName + '))' )
  else:
    r( 'library(' + libName + ')' )

def genVarName(n=1, prefix='py'):
  """Generate unique strings to use as variable names.

  The implementation uses UUIDs to avoid name clashes.

  Args:
    n (int): Number of strings to generate.
    prefix (str): Begin each generated string with this prefix.

  Returns:
    A generated string if n == 1, or a list of such if n > 1.
  """
  varNames = [None]*n
  for i in range(n):
    varNames[i] = '%s%s' % ( prefix, uuid.uuid4().hex )
  if n == 1:
    return varNames[0]
  return varNames

def pyScalarToRScalar(X, rName=None):
  """Convert a scalar to a scalar variable in the R workspace.

  Handles plus and minus infinity.

  Args:
    X: Scalar to convert.
    rName (str): Name of the R variable to which to assign the scalar.
      If None, a new variable name is auto-generated.

  Returns:
    str: The R variable to which the vector is assigned.
  """
  if rName is None:
    rName = genVarName()
  if np.isinf(X):
    if X > 0:
      r( '%s <- Inf' % rName )
    else:
      r( '%s <- -Inf' % rName )
  else:
    r( '%s <- %f' % (rName, X) )
  return rName

def pyArrayToRVector(X, rName=None, nanToNA=True):
  """Convert an array-like object to a vector in the R workspace.

  Args:
    X (array-like): Array to convert.
      Entries must be integer, float, or boolean type.
    rName (str): Name of the R variable to which to assign the vector.
      If None, a new variable name is auto-generated.
    nanToNA (bool): If True, nan in Python is converted to NA in R.
      nan in Python typically represents missing data.
      In R, missing data is represented by NA.
      (R has both NaN and NA; Python has no dedicated type for missing values.)

  Returns:
    str: The R variable to which the vector is assigned.

  Raises:
    NotImplementedError: If the type of X is unsupported.
  """
  X = np.asarray(X).ravel()

  if ( str(X.dtype).startswith('int') or
       str(X.dtype).startswith('uint') ):
    rVector = ro.IntVector(X)
  elif str(X.dtype).startswith('float'):
    rVector = ro.FloatVector(X)
  elif str(X.dtype).startswith('bool'):
    rVector = ro.BoolVector(X)
  else:
    raise NotImplementedError(
      'Only int, float, and bool are currently supported.' )

  if rName is None:
    rName = genVarName()
  r.assign(rName, rVector)

  if nanToNA:
    r( '%s[ is.nan(%s) ] <- NA' % (rName, rName) )

  return rName

def pyArrayToRMatrix(X, rName=None, nanToNA=True):
  """Convert an array-like object to a matrix in the R workspace.

  Args:
    X (array-like): 2D array to convert.
      Entries must be integer, float, or boolean type.
    rName (str): Name of the R variable to which to assign the matrix.
      If None, a new variable name is auto-generated.
    nanToNA (bool): If True, nan in Python is converted to NA in R.
      nan in Python typically represents missing data.
      In R, missing data is represented by NA.
      (R has both NaN and NA; Python has no dedicated type for missing values.)

  Returns:
    str: The R variable to which the matrix is assigned.

  Raises:
    NotImplementedError: If the type of X is unsupported.
  """
  X = np.asarray(X)
  if rName is None:
    rName = genVarName()
  pyArrayToRVector( X.ravel(order='F'), rName, nanToNA )
  r( rName + ' <- matrix(' +
    'data=' + rName + ', ' +
    'nrow=' + repr(X.shape[0]) + ', ' +
    'ncol=' + repr(X.shape[1]) + ')' )
  return rName

def pyArrayListToRMatrixList(X, rName=None, nanToNA=True):
  """Move a list of arrays from Python to R.

  Args:
    X (list): The list of array-like objects to convert.
      1D arrays become R vectors, 2D arrays become R matrices.
      Array entries must be integer, float, or boolean type.
    rName (str): Name of the R variable to which to assign the list.
      If None, a new variable name is auto-generated.
    nanToNA (bool): If True, nan in Python is converted to NA in R.
      nan in Python typically represents missing data.
      In R, missing data is represented by NA.
      (R has both NaN and NA; Python has no dedicated type for missing values.)

  Returns:
    str: The R variable to which the list is assigned.

  Raises:
    NotImplementedError: If the type of any array is unsupported.
  """
  rTemp = []
  for Xn in X:
    Xn = np.asarray(Xn)
    if Xn.ndim == 1:
      rTemp.append( pyArrayToRVector(Xn, nanToNA=nanToNA) )
    else:
      rTemp.append( pyArrayToRMatrix(Xn, nanToNA=nanToNA) )
  if rName is None:
    rName = genVarName()
  r( '%s <- list(%s)' % (rName, ','.join(rTemp)) )
  rRemove(rTemp)
  return rName

def pyArrayFromRVector(rName):
  """Convert a vector in the R workspace to a numpy array.

  Args:
    rName (str): Variable in the R workspace.

  Returns:
    array: The numpy representation of the vector.
  """
  if r('typeof(%s)' % rName)[0] == 'logical':
    return np.array( r(rName), dtype=bool )
  return np.array( r(rName) )

def pyArrayFromRMatrix(rName):
  """Convert a matrix in the R workspace to a numpy array.

  Converts anything that can be coerced to a matrix by as.matrix.

  Args:
    rName (str): Variable in the R workspace.

  Returns:
    array: The numpy representation of the matrix.
  """
  if r('typeof(%s)' % rName)[0] == 'logical':
    return np.array( r('as.matrix(%s)' % rName), dtype=bool )
  return np.array( r('as.matrix(%s)' % rName) )

def pyArrayListFromRMatrixList(rName):
  """Move a list of arrays from R to Python.

  Args:
    rName (str): Variable in the R workspace holding a list of matrices.

  Returns:
    list: Numpy representations of the list of matrices.
  """
  n = r( 'length(%s)' % rName )[0]
  result = [None]*n
  for i in range(n):
    # +1 because arrays in R are 1-based.
    result[i] = pyArrayFromRMatrix( '%s[[%d]]' % (rName, i+1) )
  return result

def rRemove(rVar):
  """Remove variables from the R workspace.

  Args:
    rVar: Name of a variable in the R workspace, or a list/tuple of such.
  """
  if isinstance(rVar, str):
    r( "remove(" + rVar + ")" )
  else:
    r( "remove(" + ", ".join(rVar) + ")" )
