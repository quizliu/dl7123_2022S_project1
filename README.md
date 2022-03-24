# dl7123_2022S_project1
## description
This is project1's repo for dl7123.

project1.py is a command line tool for building and training a model. No other codes needed.

project1.pt is the trained model.

plot_essay.py and results are used for plot figures in our report.


## usage
Dependency libs: `torch, torchvision, numpy, matplotlib.pyplot, tqdm`

To build and train our final model: `python project1.py -e 400`

To see the help message: `python project1.py -h`

## result

|  epoch   | train loss  |  test loss   | test accuracy |
| :----:  | :----:  |  :----:  | :----:  |
| 400  |0.8672| 0.9989  | 0.9414 |

Note: the model used label smoothing so loss values are higher than the normal ones.
