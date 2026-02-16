#pycaret library is used to implement explainable ai with fewer lines of code

from pycaret.datasets import get_data
from pycaret.classification import *
data = get_data('diabetes')

s = setup(data,target='Class variable',session_id=43)


best_model = compare_models()
print(best_model)

print("Creating model  Random Forest,as lrp is good but random forest is complex so it is selected")
model = create_model('rf')

print('Generating summary plot (which show which column is important for the whole dataset)')
interpret_model(model,plot='summary')

print('Generating reason plot,it tells the reason for a specific patient')
interpret_model(model,plot='reason')
