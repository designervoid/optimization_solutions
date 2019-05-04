cur_x = -0.5  # The algorithm starts at x=3
cur_y = -0.5
rate = 1  # Learning rate
precision = 0.00001  # This tells us when to stop the algorithm
previous_step_size_x = 1
previous_step_size_y = 1
max_iters = 100000  # maximum number of iterations
iters = 0  # iteration counter


def grad_x(x1, y1):
    return 1.0e-6*(-2.0*x1*(x1**2 - 1.0e-6) - 1.0e-6*x1/y1**2)*(1.0e-6*x1**2/y1**2 + (x1**2 - 1.0e-6)**2)**(-1.5)


def grad_y(x2, y2):
    return 1.0e-12*x2**2*(1.0e-6*x2**2/y2**2 + (x2**2 - 1.0e-6)**2)**(-1.5)/y2**3


while previous_step_size_x > precision and previous_step_size_y > precision and iters < max_iters:
    prev_x = cur_x  # Store current x value in prev_x
    prev_y = cur_y  # Store current y value in prev_y
    cur_x = cur_x + rate * grad_x(prev_x, prev_y)  # Grad descent x
    cur_y = cur_y + rate * grad_y(prev_x, prev_y)  # Grad descent y
    previous_step_size_x = abs(cur_x + prev_x)  # Change in x
    previous_step_size_y = abs(cur_y + prev_y)  # Change in y
    iters = iters + 1  # iteration count
    print("Iteration", iters, "\nX, Y value is", cur_x, cur_y)  # Print iterations

print("The local maximum occurs at", "x: ", cur_x, "y: ", cur_y) # Print result

