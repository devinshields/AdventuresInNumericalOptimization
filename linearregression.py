#!/usr/bin/python
''' this module solves an example of a linear least squares regression problem via gradient descent'''


import numpy as np


def least_squares_via_gradient_descent(A, b, max_gradient_norm=10**-6):
  ''' http://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_linear_system
        Minimize: ||A*x - b||
        Where   : A is a known, N-by-N matrix
                : b is a known, N vector
                : x is an unknown N vector'''

  # check that A is a square matrix, b size matches A, and that the system is over-determined
  assert isinstance(A.shape[0], (int, long)) and isinstance(A.shape[1], (int, long))
  assert len(A.shape) == 2
  assert A.shape[0] == b.shape[0]
  assert b.shape[1] == 1
  assert A.shape[0] >= A.shape[1]
  
  def f(x, A=A, b=b):
    ''' optimization value of |A*x - b| at x'''
    return np.linalg.norm(A*x-b)
  
  def f_grad(x, A=A, b=b):
    ''' gradient of |A*x - b| at x'''
    return 2*A.T*(A*x - b)

  def line_search(x, x_delta, f=f, f_grad=f_grad, t=1, beta=.9):
    ''' define a back-tracking line search tool. 'x_delta' represents search direction. '''
    while f(x + t * x_delta) > f(x + beta * t * x_delta):
      t = beta * t
    return beta * t

  # start at at x = 0, any guess will do
  x_guess = np.matrix(np.zeros(A.shape[1])).T

  # iterate until optimality
  while np.linalg.norm(f_grad(x_guess)) > max_gradient_norm:
    ''' see Boyde, convex optimization, page 464. this line search uses a
        very simple greedy gradient descent'''
    x_delta  = -1 * f_grad(x_guess)
    t        = line_search(x_guess, x_delta)
    x_guess += t * x_delta
  return x_guess


def main():
  '''  '''

  # define test data
  np.random.seed(seed=42)

  n = 5
  A = np.matrix([np.ones(n),range(n)]).T
  x = np.matrix([5, 10]).T
  b = A*x + np.matrix(np.random.normal(size=n)).T

  # call the optimization, print the results, compare with a real solution
  x_opt = least_squares_via_gradient_descent(A, b)

  print
  print 'Homegrown Gradient Descent (GD) Method'
  print 'GD - Best Fit Params:'
  print x_opt
  print
  print 'GD - Optimal Value for |A*x - b|:'
  print np.linalg.norm(A*x_opt-b)
  print
  print 'GD - Gradient at Solution:'
  print 2*A.T*(A*x_opt - b)
  print
  print 'GD - Gradient Norm at Solution:'
  print np.linalg.norm(2*A.T*(A*x_opt - b))
  print 
  print

  # the real solution found the real way: (pseudo inverse of A) * b 
  x_opt = np.linalg.lstsq(A,b)[0]

  print 'Real Solution, found via linear algebra packages'
  print 'Real - Best Fit Params:'
  print x_opt
  print
  print 'Real - Optimal Value for |A*x - b|:'
  print np.linalg.norm(A*x_opt-b)
  print
  print 'Real - Gradient at Solution:'
  print 2*A.T*(A*x_opt - b)
  print
  print 'Real - Gradient Norm at Solution:'
  print np.linalg.norm(2*A.T*(A*x_opt - b))
  print 
  print
  print 'Actual solution (parameters used to generate the data):'
  print x
  print

if __name__ == '__main__':
  main()

