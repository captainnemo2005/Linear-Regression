from numpy import *
def computing_error_for_certain_point(b,m,points):
    total_error = 0
    for i in range (0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += ((y-(m*x+b))**2)
    return total_error/float(len(points))
def compute_gradient(current_b, current_m, points, learning_rate,num_interation):
    b = current_b
    m = current_m
    for i in range(num_interation):
        b,m = step_gradient(b,m,array(points),learning_rate)
    return [b, m]
def step_gradient(b_current,m_current,points,learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (-2/N)*(y - m_current*x - b_current)
        m_gradient += (-2/N)*x*(y - m_current*x - b_current)
    b_new = b_current - b_gradient*learning_rate
    m_new = m_current - m_gradient*learning_rate
    return [b_new, m_new]
def run():
    #Define our b and m
    initial_b = 0
    initial_m = 0
    learning_rate = 0.0001
    num_interation = 1000
    #getting out data from the numpy library
    points = genfromtxt("data.csv", delimiter = ",")

    print("Our b = {0}, m = {1} and error ={2} before we use gradient descent".format(initial_b,initial_m,computing_error_for_certain_point(initial_b, initial_m,points)))
    print("Calculating ......")
    [b,m] = compute_gradient(initial_b,initial_m,points,learning_rate,num_interation)
    print("With gradient descent Algorithm, b = {0}, m = {1}, error = {2}".format(b,m,computing_error_for_certain_point(b, m,points)))
if __name__ == '__main__':
    run()