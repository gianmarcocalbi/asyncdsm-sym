
class Parent:
    def __init__(self):
        self.name = "Rocco"
        self.surname = "Calbi"

    def fullname(self):
        print("Parent: " + self.get_fullname())

    def get_fullname(self):
        return self.name + " " + self.surname

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.name = "Gianmarco"

    def fullname(self):
        print("Child: " + self.get_fullname())

p = Parent()
c = Child()
p.fullname()
c.fullname()