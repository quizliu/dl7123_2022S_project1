# dl7123_2022S_project1
## description
This is project1's repo for dl7123.

project1_model.py is a command line tool for building and training a model. No other codes needed.

project1_model.pt is the trained model.

plot_essay.py and results are used for plot figures in our report.


## usage
Dependency libs: `torch>=1.10 , torchvision, numpy, matplotlib.pyplot, tqdm`

To build and train our final model: `python project1_model.py -e 400`

To see the help message describes code's parameters: `python project1_model.py -h`

To load the trained model and predict:

```python
from project1_model import project1_model
model = project1_model()
checkpoint = torch.load('project1_model.pt', map_location=torch.device("cpu"))
model.load_state_dict(checkpoint)
model.eval()
```


## result

|  epoch   | train loss  |  test loss   | test accuracy |
| :----:  | :----:  |  :----:  | :----:  |
| 400  |0.8672| 0.9989  | 0.9414 | 

Note: the model used label smoothing so loss values are higher than the normal ones.
