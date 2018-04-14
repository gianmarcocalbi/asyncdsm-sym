
class Parent:
    def __init__(self):
        self.name = "Rocco"
        self.surname = "Calbi"

    def fullname(self):
        print("Parent: " + self.get_fullname())

    def get_fullname(self):
        return self.name + " " + self.surname

class Child(Parent):
    def __init__(self, f):
        super().__init__()
        self.name = "Gianmarco"
        self.f = f

    def fullname(self):
        print("Child: " + self.get_fullname())

    def call(self):
        self.f.static()


class Static:
    def __init__(self):
        pass

    @staticmethod
    def static():
        print("static")

p = Parent()
c = Child(Static)
c.call()
