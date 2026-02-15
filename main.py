#pycaret library is used to implement explainable ai with fewer lines of code

from pycaret.datasets import get_data
from pycaret.classification import *
data = get_data('diabetes')

s = setup(data,target='Class variable',session_id=43)


