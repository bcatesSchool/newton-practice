import numpy as np

def f(x):
    return np.cos(x)

def first_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x))/h

def second_derivative(f, x, h=1e-5):
    
    second = (first_derivative(f, x + h) - first_derivative(f, x))/h
    
    return second

def optimize(f,x0,tol=1e-5):
    x = x0
    
    diff = float('inf')
    
    while diff > tol:
        
        first = first_derivative(f, x)
        second = second_derivative(f, x)
        
        if second == 0:
            continue
            
        frac = float(first / second)
        
        temp = x - frac
        
        diff = abs(temp - x)
        
        x = temp
        
    return x

def main():
    est = optimize(np.cos,2.5)
    print(est)

if __name__ == "__main__":
    main()