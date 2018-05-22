import numpy as np

class Parent:
    def __init__(self):
        self.v = 100

class Child(Parent):
    def __init__(self):
        super().__init__()
        print(self.v)

c1 = Child()
c2 = Child()
c1.v = 200
print(c1.v)
print(c2.v)