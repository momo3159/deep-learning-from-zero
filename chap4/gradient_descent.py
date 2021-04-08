def gradient_decent(f, init_x, lr=0.1, step_num=100):
  x = init_x

  for i in range(step_num):
    grad = numerical_grad(f, x)
    x -= lr * grad

  return x