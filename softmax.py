"""Softmax."""
import numpy as np
'''
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
'''
scores = np.array([3.0,1.0,0.2])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x),axis=0)

'''
def softmax(x):
    if isinstance(x[0],np.ndarray):
        res = np.empty(shape=[len(x),len(x[0])])
        for col in range(0,len(x[0])):
            sum = 0.0
            for index in range(0,len(x)):
                sum = sum+ np.exp(x[index][col])
            for index in range(0,len(x)):
                res[index][col] = np.exp(x[index][col])/sum
    else:
        res = np.empty([0])
        sum = 0.0
        for i in x:
            sum = sum+ np.exp(i)
        for i in x:
            res = np.append(res, np.exp(i)/sum)
    return res  # TODO: Compute and return softmax(x)

'''

print(softmax(scores))
print(softmax(scores*10))
print(softmax(scores/10))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
