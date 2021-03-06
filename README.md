## rlambda library

### Introduction


This library written in python is aimed to provide the next features:
- Recursive construction of lambda function objects with regular python expression
statements (similar to sympy library expressions feature)

- Creation of lambda objects that can be formatted as strings to visualize
its code.

Its implementation mostly relies on the standard python "ast" module introduced in version 2.6, but for now
its only tested on versions >= 3.6

### Setup
You can install it running the script setup.py provided
```
git clone https://github.com/Vykstorm/rlambda.git
cd rlambda
python3 setup.py install
```

### Usage

The next code is an example that shows how to construct a simple rlambda object
```python
from rlambda.abc import x
f = (x * 10) // 3
```

We created an object which is equivalent to the next lambda function:
```python
g = lambda x: (x * 10) // 3
```

They will have the same signature and evaluation result...
```python
>>> f(2)
6
>>> g(2)
6
```


#


Printing a regular lambda in your console will look something like this...

```python
>>> lambda x: x * 2
<function <lambda> at 0x7fea5954c9d8>
```

But rlambda can fetch more information about itself...
```python
from rlambda.abc import x
>>> x * 2
x : x * 2
``` 
It prints the function body and its parameters


#
Under [docs/](docs/), you can take a look at few ipython notebooks that explains in more detail, the features of this library with interactive code examples.

