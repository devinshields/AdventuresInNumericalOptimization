#!/usr/bin/python

import numpy as np
import logging


def setup_logger():
  ''' console + file logger to track experimental results '''
  logger    = logging.getLogger('adventure_log')
  logger.setLevel(logging.INFO)
  fh        = logging.FileHandler('adventure.log')
  fh.setLevel(logging.INFO)
  ch        = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(ch)
  return logger

logger = setup_logger()


def get_linear_simulation_data(model_params=[[1],[2]], variance=5, n=50, seed=42):
  ''' builds 2D simulation data for fitting experiments'''
  np.random.seed(seed=seed)
  e = np.matrix(np.random.randn(n)).T * variance
  
  x = np.matrix(model_params)
  A = np.matrix([np.ones(n), np.arange(n)]).T
  b = A*x + e
  return A, b


def solve_gradient_decent(A, b, x_guess=np.array([[0], [0]]), step_size=.00001, max_gradient_norm=.01):
  ''' http://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_linear_system '''

  def get_gradient(A, b, x):
    return 2*A.T*(A*x - b)


  grad = get_gradient(A, b, x_guess)
  while np.linalg.norm(grad) > max_gradient_norm:

    # log the current state of the optimization
    logger.debug('Optimization Value: {0}'.format(np.linalg.norm(A*x_guess-b)))
    logger.debug('Guess:', x_guess.T)
    logger.debug('Gradient:', grad.T)
    logger.debug('Gradient Norm:', np.linalg.norm(grad))

    # get the gradient at the current guess
    grad = get_gradient(A, b, x_guess)
    
    # calculate a gradient decent step & update the guess
    x_guess = x_guess - step_size*grad
  return x_guess


def main ():
  ''' builds some noisy, 2D linear sample data, then fits linear  '''

  # get some sample data
  A, b = get_linear_simulation_data()
  logger.info('Building simulation data: complete')

  # solve via the built in linear regressor
  solution = np.linalg.lstsq(A, np.asarray(b.T)[0])[0].T
  logger.info('Built-In Solver Solution: {0}'.format(solution))

  # solve via a hand-rolled gradient decent function
  solution = solve_gradient_decent(A, b)
  logger.info('Homemade Gradient Decent: {0}'.format(solution.T))

  pass

if __name__ == '__main__':
  main()
