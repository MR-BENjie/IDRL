# Red_10 game environment
<img width="500" src="https://dczha.com/files/rlcard/logo.jpg" alt="Logo" />
RLCard is a toolkit for Reinforcement Learning (RL) in card games. We use the RLCard to create the Red-10 game environment.


## installation 
Make sure you have installed **Python 3.6+** and **pip**.

Installed rlcard using pip.
```
pip3 install rlcard
```
The default installation will only include the card environments. To use PyTorch implementation of the training algorithms, run
```
pip3 install rlcard[torch]
```

Then installed with
```
cd rlcard
pip3 install -e .
pip3 install -e .[torch]
```