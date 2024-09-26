#ML #study #paper

### Definition
	"Field of study that gives computers the ability to learn without being explicitly programmed."--Arthur Samuel(1959)
[[Project Proposal]]
### Machine learning algorithms
- Supervised learning
- unsupervised learning
- (Reinforcement learning)

### Supervised learning
Learns from being given "right answers"
1. Regression: Predict a number, infinitely many possible outputs.
2. Classification: Predict categories, small number of possible outputs.

### Unsupervised Learning
Data only comes with inputs x, but not output labels y.
Algorithm has to find structure in the data.
Find something interesting in unlabeled data.
Clustering: (Group similar data points together.)
1. Google news
2. DNA microarray
3. Grouping customers
Anomaly detection:(Find unusual data points.)
Dimensionality reduction:(Compress data using fewer numbers.)

Jupyter Notebook
1. Markdown Cell
2. Code cell
shift+enter
```
variable = "right in the strings!"
print(f"f strings allow you to embed variables {variable}")
```

#### Classification Model
Predicts categories
Small number of possible outputs

### Regression Model

#### Linear Regression

Supervised learning model
Data has "right answers"

Regression model
Predicts numbers
Infinitely many possible outputs

#### Terminology
Training Set: data used to train the model
Notation: inputs: features, outputs: targets
Learning Algorithm
prediction
estimated

#### Cost Function

### Lab
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data
```
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# x_train = [1. 2.]
# y_train = [300. 500.]
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
# x_train.shape: (2,)
Number of training examples is: 2
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
# Number of training examples is: 2
i = 0 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# (x^(0), y^(0)) = (1.0, 300.0)
i = 1 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# (x^(1), y^(1)) = (2.0, 500.0)

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```

#### model
#### parameters
#### cost function
#### goal

## Gradient Descent



