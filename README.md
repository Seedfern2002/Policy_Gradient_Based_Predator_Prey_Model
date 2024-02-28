### Policy gradient based predator-prey model
The coures project *Mathematical Modeling in the Life Sciences* 2022 Spring made by Yexiang Cheng in his sophomore year. 

#### 1. File description
- [main.py](main.py) 
- [population.py](population.py) The implementation of agent(population)
- [fig_final] stores the result figures 
- [model] stores the learned weights of agent's neural network

#### 2. Install the dependency:
```bash
pip install -r requirements.txt
```

#### 3. running the program
```bash
python main.py --fig_dir [The directory to store the figures] --model_dir [The directory to store the weights of neural network]
```
the default fig_dir is './fig_final', while the default model_dir is './model'
