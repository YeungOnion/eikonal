# func1's have diff arg types
def func1_mode_a(x, y):
    return x+y
def func1_mode_b(x, y):
    return x[0]*x[1] - y

# func2's have same arg types
def func2_mode_a(u, v):
    return u-v
def func2_mode_b(u,v):
    return u*v