# function to get the root of.
# x^3 - x^2 - 15x +1
def f(x):
    return x**3 - x**2 - 15*x + 1
# the derivative of the function
# 3x^2 - 2x - 15
def df(x):
    return 3*x**2 - 2*x - 15

# apply neuton to get root 
def neuton(func,dfunc,x0,error):
    itters = 0
    x1 = x0
    x2 = x1 - func(x1)/dfunc(x1)

    while abs(x2-x1)>error:
        x1 = x2
        x2 = x1 - func(x1)/dfunc(x1)
        itters+=1

    return x2,itters

res , iters = neuton(f,df,4.5,1e-5)
print("root = "+str(res)+" and took "+str(iters)+" iterations.")
