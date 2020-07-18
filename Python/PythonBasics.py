'Ola'*4
'Ola'+'Rocks'
'Ola'.upper()
len('Ola')
len(str(20))
int(20.5)
name='Ola'
print(name)
list=[1,23,5,6,7,'name']
list[5]
len(list)
list.reverse()
print(list)
dict={'key1':2,'key2':3,'key3':'Hey'}
dict['key3']
len(dict)
6<7
yes=True
if (2>4):
    print('True')
else:
    print('False')
def hi():
    print('Hi there')
hi()
for name in list:
    print(name)
for i in range(2,10,2):
    print(i)

# Use break to come out of a block prematurely

class MyClass:
    # Attribute
    a=10
    # Method
    def f(self):
        return 'hello world'

"""Classes provide a means of bundling data and functionality together.
Creating a new class creates a new type of object, allowing new instances of that type to be made.
Each class instance can have attributes attached to it for maintaining its state.
Class instances can also have methods (defined by its class) for modifying its state."""

object=MyClass()

class Complex:
    # Constrictor
    def __init__(self,real,img):
        self.r=real
        self.i=img

Complex_number=Complex(2,3)


class Classname(object):
	def __init__(self,argument):
		self.variable=argument
		
	def method(self, method_argument):
		pass


# Exception Handling using try,except and raise
try:
	pass
except(Errorname)
	"Error Statement"
else:
	pass
	
	
# Using zip for iteration
lista=[1,3,5]
listb=[1,9,25]
for a,b in zip(lista,listb):
	print(a,b)
	
	
# Using enumerate for iteration
list=[1,3,4,9]
for i, item in enumerate(list):
  print(i,item)

for i, item in enumerate(list):
    print("Key is " + str(i))
    print("Item is " + str(item))

#List and Dictionary comprehension
squares=[i**2 for i in range(10)]
squares_dict={i:i**2 for i in range(10)}

list=[2.5,9,12]
int(list)                       # won't work
[int(item) for item in list]

[item for item in list if item > 2]

sum([item for item in list if item > 2])
sum(item for item in list if item > 2)   # Good for large datasets because of being memory efficient

new_list = [expression for member in iterable(if conditional)]

quote="Life is wonderful!"

unique_vowels=[i for i in quote if i in 'aeiou']

matrix = [[i for i in range(5)] for _ in range(6)]

import random
def get_weather_data():
    return random.randrange(90,110)

hot_temps=[temp for _ in range(20)]

hot_temps=[temp for _ in range(20) if (temp := get_weather_data()) >= 100]
# Walrus assignment operator


zip, enumerate, list comprehensions, classes, error handling

