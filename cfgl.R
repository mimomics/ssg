# Conditional fused graphical lasso (CFGL).

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

library(genlasso)
library(glasso)
library(glmnet)
library(Matrix)
library(matrixStats)

# ============================================================================
# General Functions
# ============================================================================

solveLU <- function(L, U, b)
{
  # Solve the linear system Ax = b given the decomposition A = LU.
  #
  # Args:
  #   L: The L matrix in the LU decomposition of A.
  #   U: The U matrix in the LU decomposition of A.
  #   b: RHS vector in the linear system.
  #
  # Returns:
  #   The solution x to Ax = b, where A = LU.

  # Let y = Ux; solve Ly = b for y; then solve Ux = y for x.
  y <- solve(L, b)
  x <- solve(U, y)
  return(x)
}

frobNormND <- function(A)
{
  # Frobenius norm of A generalised to arrays.
  #
  # Args:
  #   A: Array whose norm is required.
  #
  # Returns:
  #   Generalised Frobenius norm: root sum of squares of all elements.
  return( sqrt(sum(A**2)) )
}

logdet <- function(A)
{
  # Compute log(det(A)) efficiently.
  return( 2*log(prod(diag(chol(A)))) )
}

mvnLogPdf <- function(X, mu, K)
{
  # Log pdf of a multivariate Gaussian.
  # Takes a precision (unlike most implementation, which take a covariance).
  X0 = X - mu
  L = t(chol(K))
  X0L = X0 %*% L
  logDetL = sum( log(diag(L)) )
  lpdf = logDetL - 0.5*( nrow(K)*log(2*pi) + rowSums(X0L^2) )
  lpdf[ is.infinite(rowSums(X)) ] = -Inf
  return(lpdf)
}

# ============================================================================
# Support Functions
# ============================================================================

buildPenaltyArray <- function(sPen, fPen, nRow, nCol, nClass)
{
  # Construct a penalty array from scalars.
  #
  # Build a 4-dimensional penalty array as used by fmvl, fgl,
  # and cfgl, when the sparsity and fusion penalties are scalars.
  #
  # Args:
  #   sPen: Sparsity penalty.
  #   fPen: Fusion penalty.
  #   nRow: Number of rows of the matrix to penalise.
  #   nRow: Number of columns of the matrix to penalise.
  #   nClass: Number of classes.

  delta <- array(fPen, dim=c(nRow, nCol, nClass, nClass) )
  for(i in 1:nClass) delta[,,i,i] <- sPen
  return(delta)
}

# ============================================================================
# Fused Multivariate Lasso (FMVL)
# ============================================================================

fmvl <- function(
  N, S, H, K, delta, B,
  rho=1, maxIter=100, tolAbs=1e-3, tolRel=1e-3,
  useSVD=FALSE,
  objective=FALSE )
{
  # Fused Multivariate Lasso.
  #
  # S, H, K, B, are 3-dimensional arrays, each containing one
  # matrix per class. The 3rd dimension corresponds to the class.
  # delta is a 4-dimensional array.
  #
  # ADMM is used to optimise the fmvl objective. ADMM is described here:
  # https://web.stanford.edu/class/ee367/reading/admm_distr_stats.pdf
  # Boyd, Stephen, et al.
  # "Distributed optimization and statistical learning via the alternating
  # direction method of multipliers."
  # Foundations and Trends in Machine Learning 3.1 (2011): 1-122.
  #
  # See the paper for details of ADMM, and in particular for
  # the meaning of the relative and absolute tolerance parameters.
  #
  # Args:
  #   N (vector): N[c] is the number of training data points for class c.
  #   S (array): S[,,c] = t(X[[c]]) * X[[c]].
  #   H (array): H[,,c] = t(X[[c]]) * Y[[c]] * K[,,c].
  #   K (array): K[,,c] is the precision matrix for class c.
  #   delta (array): L1 penalty strengths.
  #     Either a 2-element vector c(sparsity_penalty, fusion_penalty),
  #     or a 4-dimensional array in which
  #     delta[,,c,d] is the matrix of fusion penalties between classes c and d.
  #     delta[,,c,c] is the matrix of sparsity penalties for class c.
  #   B (array): B[,,c] is the initial matrix of the regression coefficients
  #     for class c.
  #   rho (double): ADMM penalty parameter.
  #   maxIter (int): Maximum number of iterations.
  #   tolAbs (double): Absolute tolerance.
  #   tolRel (double): Relative tolerance.
  #   useSVD (logical): The genlasso package is used to solve step 2 of the
  #     ADMM algorithm. This corresponds to the svd argument of genlasso.
  #     If TRUE, genlasso is slower but more stable.
  #   objective (logical): Compute the objective function at each iteration.
  #
  # Returns:
  #   list: Class fmvl. Has elements
  #     B: Learned regression parameters.
  #     obj: Objective function at each iteration, if requested.

  nClass <- dim(B)[3]
  pX <- dim(B)[1]
  pY <- dim(B)[2]

  if (is.vector(delta))
    delta <- buildPenaltyArray(delta[1], delta[2], pX, pY, nClass)

  tolAbsScaled <- sqrt( pX * pY * nClass ) * tolAbs

  kpL <- list()
  kpU <- list()
  vecScaledH <- list()
  for(i in 1:nClass)
  {
    a <- 2 / (N[i] * rho)

    kp <- a * kronecker( K[,,i], S[,,i] ) + diag(1, pX*pY)
    decomp <- expand( lu(kp) )
    kpL[[i]] <- decomp$L
    kpU[[i]] <- decomp$U

    vecScaledH[[i]] <- a * c( H[,,i] )
  }

  I <- diag(nClass)
  rhoInv <- 1/rho

  Z <- B
  U <- array( 0, dim=dim(B) )

  if (objective)
  {
    obj <- list()
    obj[[1]] <- fmvlObjective(B, N, S, H, K, delta)
  }

  for(iter in 1:maxIter)
  {
    # ADMM step 1. Update B.
    for(i in 1:nClass)
    {
      B[,,i] <- matrix(
        solveLU( kpL[[i]], kpU[[i]], vecScaledH[[i]] + c(Z[,,i]) - c(U[,,i]) ),
        nrow=pX )
    }

    # ADMM step 2. Update Z.
    Zold <- Z
    for(i in 1:pX)
    {
      for(j in 1:pY)
      {
        y <- B[i,j,] + U[i,j,]

        # Form the D matrix used by genlasso such that
        # sparsity and fusion penalties are applied.

        # Sparsity penalties.
        D <- diag(diag( delta[i,j,,] ))

        # Fusing penalties. Add a row to D for each class pair.
        for( m in 1:(nClass-1) )
        {
          for( n in (m+1):nClass )
          {
            pen <- delta[i,j,m,n]
            rowD <- rep(0, nClass)
            rowD[m] <- pen
            rowD[n] <- -pen
            D <- rbind(D, rowD)
          }
        }

        objPre <- genLassoObjective(y, I, c(Z[i,j,]), rhoInv, D)

        # Passing D as a sparse matrix here does not work because genlasso
        # appears to require that a sparse D not have more rows than columns.
        glResult <- genlasso(y, D=D, minlam=rhoInv, svd=useSVD)
        Z[i,j,] <- coef(glResult, lambda=rhoInv)$beta

        # genlasso has some instability issues: sometimes, its objective
        # function increases. When this happens, we ignore genlasso's output,
        # and leave Z unchanged.
        objPost <- genLassoObjective(y, I, c(Z[i,j,]), rhoInv, D)
        if (objPre-objPost < 0)
          Z[i,j,] <- Zold[i,j,]
      }
    }

    # ADMM step 3. Update U.
    U <- U + B - Z

    if (objective)
      obj[[iter+1]] <- fmvlObjective(B, N, S, H, K, delta)

    # Check the ADMM stopping criterion.

    # Primal and dual tolerances.
    tolPrimal <- tolAbsScaled + tolRel * max( frobNormND(B), frobNormND(Z) )
    tolDual   <- tolAbsScaled + tolRel * rho * frobNormND(U)

    # Primal and dual residual norms.
    resNormPrimal <- frobNormND(B - Z)
    resNormDual   <- frobNormND( rho*(Z - Zold) )

    if ( (resNormPrimal <= tolPrimal) &&
         (resNormDual   <= tolDual) )
      break
  }

  result <- structure(list(B=B), class="fmvl")
  if (objective)
    result$obj <- unlist(obj)
  return(result)
}

genLassoObjective <- function(y, X, b, lambda, D)
{
  # Compute genlasso's objective function.

  y0 <- y - X %*% b
  return( (sum(y0 * y0)/2) + lambda * sum(abs(D %*% b)) )
}

fmvlObjective <- function(B, N, S, H, K, delta)
{
  # Compute the objective function that fmvl minimises, up to a constant.
  # See the fmvl function for argument descriptions.

  nClass <- length(N)
  obj <- 0
  for(i in 1:nClass)
  {
    # Log likelihood terms.
    BSB <- t(B[,,i]) %*% S[,,i] %*% B[,,i]
    obj <- obj + ( sum(BSB*K[,,i]) - 2*sum(B[,,i]*H[,,i]) ) / N[i]
    obj <- obj - logdet(K[,,i])

    # Sparsity penalties.
    obj <- obj + sum( delta[,,i,i] * abs(B[,,i]) )
  }

  # Fusion penalties.
  for(i in 1:(nClass-1))
  {
    for(j in (i+1):nClass)
    {
      obj <- obj + sum( delta[,,i,j] * abs(B[,,i] - B[,,j]) )
    }
  }

  return(obj)
}

# ============================================================================
# Fused Graphical Lasso (FGL)
# ============================================================================

fgl <- function(
  N, S, K, delta,
  rho=1, maxIter=100, tolAbs=1e-3, tolRel=1e-3,
  objective=FALSE )
{
  # Fused Graphical Lasso.
  #
  # The fused graphical lasso is introduced in the following paper:
  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4012833/
  # Danaher, Patrick, Pei Wang, and Daniela M. Witten.
  # "The joint graphical lasso for inverse covariance estimation across
  # multiple classes."
  # Journal of the Royal Statistical Society: Series B
  # (Statistical Methodology) 76.2 (2014): 373-397.
  #
  # S and K are 3-dimensional arrays, each containing one matrix per class.
  # The 3rd dimension corresponds to the class. delta is a 4-dimensional array.
  #
  # Args:
  #   N (vector): N[c] is the number of training data points for class c.
  #   S (array): S[,,c] = t(Y[[c]]) %*% Y[[c]],
  #     where Y[[c]] is the data matrix for class c.
  #   K (array): K[,,c] is the initial precision matrix for class c.
  #   delta (array): L1 penalty strengths.
  #     Either a 2-element vector c(sparsity_penalty, fusion_penalty),
  #     or a 4-dimensional array in which
  #     delta[,,c,d] is the matrix of fusion penalties between classes c and d.
  #     delta[,,c,c] is the matrix of sparsity penalties for class c.
  #   rho (double): ADMM penalty parameter.
  #   maxIter (int): Maximum number of iterations.
  #   tolAbs (double): Absolute tolerance.
  #   tolRel (double): Relative tolerance.
  #   objective (logical): Compute the objective function at each iteration.
  #
  # Returns:
  #   list: Class fgl. Has elements
  #     K: Learned precision matrices.
  #     obj: Objective function at each iteration, if requested.

  nClass <- dim(K)[3]
  pK <- dim(K)[1]

  if (is.vector(delta))
    delta <- buildPenaltyArray(delta[1], delta[2], pK, pK, nClass)

  I <- diag(nClass)
  tolAbsScaled <- sqrt(nClass*pK*pK) * tolAbs

  Z <- K
  U <- array( 0, dim=dim(K) )

  if (objective)
  {
    obj <- list()
    obj[[1]] <- fglObjective(N, S, K, delta)
  }

  for(iter in 1:maxIter)
  {
    # ADMM step 1. Update K.
    for(i in 1:nClass)
    {
      ed <- eigen( S[,,i]/N[i] + rho*( U[,,i] - Z[,,i] ), symmetric=TRUE )
      V <- ed$vectors
      D <- ed$values
      D <- diag( (1/(2*rho))*(sqrt(D**2 + 4*rho) - D) )
      K[,,i] <- symmpart( V %*% D %*% t(V) )
    }

    # ADMM step 2. Update Z.
    Zold <- Z
    for(i in 1:pK)
    {
      for(j in i:pK)
      {
        y <- K[i,j,] + U[i,j,]

        # Form the D matrix used by genlasso such that
        # sparsity and fusion penalties are applied.

        # Sparsity penalties.
        D <- diag(diag( delta[i,j,,] ))

        # Fusing penalties. Add a row to D for each class pair.
        for( m in 1:(nClass-1) )
        {
          for( n in (m+1):nClass )
          {
            pen <- delta[i,j,m,n]
            rowD <- rep(0, nClass)
            rowD[m] <- pen
            rowD[n] <- -pen
            D <- rbind(D, rowD)
          }
        }

        # Passing D as a sparse matrix here does not work because genlasso
        # appears to require that a sparse D not have more rows than columns.
        glResult <- genlasso(y, I, D, minlam=1/rho)
        Z[i,j,] <- coef(glResult, lambda=1/rho)$beta
        Z[j,i,] <- Z[i,j,]
      }
    }

    # ADMM step 3. Update U.
    U <- U + K - Z

    if (objective)
      obj[[iter+1]] <- fglObjective(N, S, K, delta)

    # Check the ADMM stopping criterion.
    # We do not take symmetry into account when computing the
    # derivatives with respect to K and Z. (This is the same as in the
    # JGL paper, and does not change the result of the optimisation.)
    # So we also ignore symmetry when computing the stopping criteria.

    # Primal and dual tolerances.
    tolPrimal <- tolAbsScaled + tolRel * max( frobNormND(K), frobNormND(Z) )
    tolDual   <- tolAbsScaled + tolRel * rho * frobNormND(U)

    # Primal and dual residual norms.
    resNormPrimal <- frobNormND(K - Z)
    resNormDual   <- frobNormND( rho*(Z - Zold) )

    if ( (resNormPrimal <= tolPrimal) &&
         (resNormDual   <= tolDual) )
      break
  }

  result <- structure(list(K=K), class="fgl")
  if (objective)
    result$obj <- unlist(obj)
  return(result)
}

fglObjective <- function(N, S, K, delta)
{
  # Compute the objective function that fgl minimises.
  # See the fgl function for argument descriptions.

  nClass <- length(N)
  obj <- 0
  for(i in 1:nClass)
  {
    # Log likelihood.
    obj <- obj + sum( S[,,i] * K[,,i] ) / N[i] - logdet(K[,,i])

    # Sparsity penalties.
    obj <- obj + sum( delta[,,i,i] * abs(K[,,i]) )
  }

  # Fusion penalties.
  for(i in 1:(nClass-1))
  {
    for(j in (i+1):nClass)
    {
      obj <- obj + sum( delta[,,i,j] * abs(K[,,i] - K[,,j]) )
    }
  }

  return(obj)
}

# ============================================================================
# Conditional Fused Graphical Lasso (CFGL)
# ============================================================================

cfgl <- function(
  X, Y, B=NULL, K=NULL, intercept=TRUE,
  deltaB, rhoB, maxIterB=100, tolAbsB=1e-3, tolRelB=1e-3,
  deltaK, rhoK, maxIterK=100, tolAbsK=1e-3, tolRelK=1e-3,
  maxIter=100, tol=1e-5,
  useSVD=FALSE,
  logLikelihood=FALSE, objective=FALSE )
{
  # Conditional Fused Graphical Lasso
  #
  # Alternates steps of fmvl (for the regression coefficients) and fgl
  # (for the precision matrices). Both fmvl and fgl use ADMM optimisation.
  # Arguments ending in B refer to fmvl/regression;
  # arguments ending in K refer to fgl/precisions.
  #
  # Args:
  #   B (array): B[,,c] is the initial matrix of regression coefficients
  #     for class c.
  #   K (array): K[,,c] is the initial precision matrix for class c.
  #   intercept (logical): Whether to fit an intercept in the regressions.
  #   deltaB (array): L1 penalty strengths for the regression coefficients.
  #     Either a 2-element vector c(sparsity_penalty, fusion_penalty),
  #     or a 4-dimensional array in which
  #     deltaB[,,c,d] is the fusion penalty matrix between classes c and d.
  #     deltaB[,,c,c] is the sparsity penalty matrix for class c.
  #   rhoB (double): ADMM penalty parameter for fmvl.
  #   maxIterB (int): Maximum number of iterations in each fmvl call.
  #   tolAbsB (double): Absolute tolerance for fmvl.
  #   tolRelB (double): Relative tolerance for fmvl.
  #   deltaK (array): L1 penalty strengths for the precision matrices.
  #     Either a 2-element vector c(sparsity_penalty, fusion_penalty),
  #     or a 4-dimensional array in which
  #     deltaK[,,c,d] is the fusion penalty matrix between classes c and d.
  #     deltaK[,,c,c] is the sparsity penalty matrix for class c.
  #   rhok (double): ADMM penalty parameter for fgl.
  #   maxIterK (int): Maximum number of iterations in each fgl call.
  #   tolAbsK (double): Absolute tolerance for fgl.
  #   tolRelK (double): Relative tolerance for fgl.
  #   maxIter (int): Maximum number of iterations of the outer loop.
  #   tol (double): Tolerance. The algorithm terminates when the maximum
  #     absolute change in any parameter falls below this value.
  #   useSVD (logical): The genlasso package is used to solve step 2 of the
  #     fmvl ADMM algorithm. This corresponds to the svd argument of genlasso.
  #     If TRUE, genlasso is slower but more stable.
  #   logLikelihood (logical): Compute the log likelihood of the training data
  #     at each iteration.
  #   objective (logical): Compute the objective function at each iteration.
  #
  # Returns:
  #   list: Class cfgl. Has elements
  #     B: Learned regression parameters.
  #     K: Learned precision matrices.
  #     priors: Vector of class priors.
  #     logLik: Log likelihood at each iteration, if requested.
  #     obj: Objective function at each iteration, if requested.

  nClass <- length(X)
  pX <- ncol( X[[1]] )
  pY <- ncol( Y[[1]] )
  if (intercept) pX <- pX + 1

  if (is.vector(deltaB))
    deltaB <- buildPenaltyArray(deltaB[1], deltaB[2], pX, pY, nClass)
  if (is.vector(deltaK))
    deltaK <- buildPenaltyArray(deltaK[1], deltaK[2], pY, pY, nClass)

  stopifnot( dim(deltaB)[1] == pX )

  N <- rep(0, nClass)
  XX <- array(0, dim=c(pX, pX, nClass))
  XY <- array(0, dim=c(pX, pY, nClass))
  for(i in 1:nClass)
  {
    N[i] <- nrow( X[[i]] )
    if (intercept)  X[[i]] <- cbind( rep(1, N[i]), X[[i]] )
    XX[,,i] <- t( X[[i]] ) %*% X[[i]]
    XY[,,i] <- t( X[[i]] ) %*% Y[[i]]
  }

  if ( is.null(B) )
  {
    B <- array( 0, dim=c(pX, pY, nClass) )
    for(i in 1:nClass)
    {
      # Initialise B using independent lasso estimates.
      for(j in 1:pY)
      {
        r <- glmnet(
          x=if (intercept) X[[i]][,2:pX] else X[[i]],
          y=Y[[i]][,j],
          family="gaussian",
          intercept=intercept )
        B[,j,i] <- as.matrix( coef(r, s=mean(deltaB[,,i,i])) )
      }
    }
  }
  if ( is.null(K) )
  {
    K <- array( 0, dim=c(pY, pY, nClass) )
    for(i in 1:nClass)
    {
      # Initialise the precisions using graphical lasso.
      Yi0 <- Y[[i]] - X[[i]] %*% B[,,i]
      Ki <- glasso( cov(Yi0), deltaK[,,i,i] )$wi
      K[,,i] <- ( Ki + t(Ki) ) / 2
    }
  }

  if (logLikelihood)
  {
    llh <- list()
    model <- structure(list(B=B, K=K), class="cfgl")
    llh[[1]] <- logLik(model, X, Y)
  }
  if (objective)
  {
    obj <- list()
    model <- structure(list(B=B, K=K), class="cfgl")
    obj[[1]] <- cfglObjective(model, X, Y, deltaB, deltaK)
  }

  H  <- array(0, dim=c(pX, pY, nClass))
  YY0 <- array(0, dim=c(pY, pY, nClass))

  for(iter in 1:maxIter)
  {
    print( sprintf('Iteration %d of %d.', iter, maxIter) )

    prevB <- B
    prevK <- K

    # Update B.
    for(i in 1:nClass)
      H[,,i] <- XY[,,i] %*% K[,,i]
    B <- fmvl(
      N, XX, H, K, deltaB, B,
      rhoB, maxIterB, tolAbsB, tolRelB,
      useSVD=useSVD, objective=FALSE )$B
    maxDiff <- max(abs(B - prevB))

    # Update K.
    for(i in 1:nClass)
    {
      Yi0 <- Y[[i]] - X[[i]] %*% B[,,i]
      YY0[,,i] <- t(Yi0) %*% Yi0
    }
    K <- fgl(N, YY0, K, deltaK, rhoK, maxIterK, tolAbsK, tolRelK)$K
    maxDiff <- max( maxDiff, max(abs(K - prevK)) )

    if (logLikelihood)
    {
      model <- structure(list(B=B, K=K), class="cfgl")
      llh[[iter+1]] <- logLik(model, X, Y)
    }
    if (objective)
    {
      model <- structure(list(B=B, K=K), class="cfgl")
      obj[[iter+1]] <- cfglObjective(model, X, Y, deltaB, deltaK)
    }

    # Stop when the maximum change in any parameter is below tolerance.
    print( sprintf('Maximum parameter change = %f', maxDiff) )
    if (maxDiff < tol) break
  }

  result <- structure(list(B=B, K=K, priors=N/sum(N)), class="cfgl")
  if (logLikelihood)
    result$logLik <- unlist(llh)
  if (objective)
    result$obj <- unlist(obj)
  return(result)
}

logLik.cfgl <- function(model, X, Y)
{
  # Compute log probability of the outputs given the inputs and classes.
  #
  # Args:
  #   model: Return value from cfgl.
  #   X: List of inputs, as for cfgl.
  #   Y: List of outputs, as for cfgl.
  #
  # Returns:
  #   Object of class "logLik", a scalar.
  #   Has one attribute, df, the number of estimated parameters in the model.

  nClass <- dim(model$B)[3]
  llh <- 0
  for(i in 1:nClass)
  {
    # Augment inputs with a column of ones if B has an intercept row.
    if ( ncol(X[[i]]) + 1 == dim(model$B)[1] )
      X[[i]] <- cbind( rep(1, nrow(X[[i]])), X[[i]] )

    llh <- llh + sum(mvnLogPdf(Y[[i]], X[[i]] %*% model$B[,,i], model$K[,,i]))
  }

  df <- prod( dim(model$B) ) + prod( dim(model$K) )
  return( structure(llh, class='logLik', df=df) )
}

cfglPredict <- function(model, X, Y)
{
  # Compute class probabilities for a set of data points.
  #
  # Args:
  #   model: Return value from cfgl.
  #   X: Matrix of inputs.
  #   Y: Matrix of outputs.
  #
  # Returns:
  #   Matrix, same number of rows as X and Y, columns correspond to classes.
  nClass <- dim(model$B)[3]
  N = nrow(X)

  # Augment inputs with a column of ones if B has an intercept row.
  if ( ncol(X) + 1 == dim(model$B)[1] )
    X <- cbind(rep(1, N), X)

  lpr = log(model$priors)
  lf = matrix(nrow=nClass, ncol=N)
  for(i in 1:nClass)
  {
    lf[i,] = mvnLogPdf(Y, X %*% model$B[,,i], model$K[,,i]) + lpr[i]
  }
  lZ = apply(lf, 2, logSumExp)
  return( t(exp(sweep(lf, 2, lZ))) )
}

cfglObjective <- function(model, X, Y, deltaB, deltaK)
{
  # Compute the objective function that cfgl minimises.
  #
  # For more information about the arguments, see the cfgl function.
  #
  # Args:
  #   model: Return value from cfgl.
  #   X: List of inputs.
  #   Y: List of outputs.
  #   deltaB: Regression parameter penalties.
  #   deltaK: Precision parameter penalties.
  #
  # Returns:
  #   The objective function value.

  nClass <- dim(model$B)[3]
  obj <- 0

  for(i in 1:nClass)
  {
    cholK <- chol( model$K[,,i] )

    # Trace term.
    Y0 <- Y[[i]] - X[[i]] %*% model$B[,,i]
    A <- Y0 %*% t(cholK)
    obj <- obj + sum(A*A) / nrow(Y0)

    # Log determinant term.
    obj <- obj - 2*log(prod(diag(cholK)))

    # Sparsity penalty terms.
    sPenB <- sum( deltaB[,,i,i] * abs(model$B[,,i]) )
    sPenK <- sum( deltaK[,,i,i] * abs(model$K[,,i]) )
    obj <- obj + sPenB + sPenK
  }

  # Fusion penalty terms.
  for(i in 1:(nClass-1))
  {
    for(j in (i+1):nClass)
    {
      fPenB <- sum( deltaB[,,i,j] * abs(model$B[,,i] - model$B[,,j]) )
      fPenK <- sum( deltaK[,,i,j] * abs(model$K[,,i] - model$K[,,j]) )
      obj <- obj + fPenB + fPenK
    }
  }

  return(obj)
}

# ============================================================================
# CFGL Penalty Tuning
# ============================================================================

cfglApproxAIC <- function(model, X, Y)
{
  # Compute an approximate AIC for optimal penalty selection.
  #
  # This is analogous to the method used in the Section 6 of the JGL paper.
  #
  # Args:
  #   model: Return value from cfgl.
  #   X: List of inputs, as for cfgl.
  #   Y: List of outputs, as for cfgl.
  #
  # Returns:
  #   The approximate AIC.
  nClass <- dim(model$B)[3]
  pY <- dim(model$K)[1]
  aic <- 0
  for(i in 1:nClass)
    # E is the number of free parameters.
    E <- sum( model$K[,,i] != 0 ) +
         sum( model$K[,,i][ upper.tri(model$K[,,i]) ] != 0 ) + pY
        aic <- aic + logLik(model, X, Y) + E
  return(aic)
}

cfglTune <- function(
  X, Y, B=NULL, K=NULL, intercept=TRUE,
  deltaB, rhoB, maxIterB=100, tolAbsB=1e-3, tolRelB=1e-3,
  deltaK, rhoK, maxIterK=100, tolAbsK=1e-3, tolRelK=1e-3,
  maxIter=100, tol=1e-5, useSVD=FALSE )
{
  # Train a cfgl model with automatic penalty selection via approximate AIC.
  #
  # Arguments are the same as cfgl, except that deltaB and deltaK are lists.
  # They have the same number of elements;
  # corresponding pairs are the values to test with AIC.

  aicBest   <- Inf
  modelBest <- NULL
  iBest     <- NULL

  for ( i in 1:length(deltaB) )
  {
    model <- cfgl(
      X, Y, B, K, intercept,
      deltaB[[i]], rhoB, maxIterB, tolAbsB, tolRelB,
      deltaK[[i]], rhoK, maxIterK, tolAbsK, tolRelK,
      maxIter, tol, useSVD )
    aic <- cfglApproxAIC(model, X, Y)
    if (aic < aicBest)
    {
      aicBest   <- aic
      modelBest <- model
      iBest     <- i
    }
  }

  return( list(model=modelBest,
               deltaB=deltaB[[i]],
               deltaK=deltaK[[i]]) )
}
