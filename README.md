# 046211 modern optimizers for multiclass tasks
Welcome to the to our project repository.
This repository houses the code and resources for checking Adan, Madgrad, ScheduleFree new optimizers performances on Audio and image classifications tasks.
Our main goal is to check which optimizer perform the best for those task validating their articles.
Every optimizer show state of the art performance for many common tasks.

## Overview
In the recent years, new optimization methods arise.
while Adam is well known and used, different optimization technique claims that they are superior for specific cases.

- Adan - (https://arxiv.org/pdf/2208.06677) uses nestrov acceleration differently show Sota results for vision transformer, bert and gpt and promsing results for RL method.
    further download and documentation is here: (https://github.com/lucidrains/Adan-pytorch/tree/main)

- Madgrad - (https://arxiv.org/pdf/2101.11075) - uses dual-averaging method which average the gradients over time. With that it should be more stable and converge better 
    further download and documentation is here: (https://github.com/facebookresearch/madgrad)

- Schedule free - (https://arxiv.org/pdf/2405.15682) Is a new scheduling technique using many optimization technique as adaptability, averaging, nestrov acceleration.

    It doesnt need scheduling time to be set in order reach very good performances vs other scheduling technique. 

## Methodology
In this project, we tested the optimizers credability for NLP task and vision task.
We tested SGD, Adam, Madgrad, Adan and ScheduleFree with the 2 tasks and checked if loss and accuracy is behave as the article suggest.

We chose the following tasks:
-  ECS-50 Audio Dataset for multiclass NLP classification using Hubert
-  image............................ 


## Data

### ECS-50:
The datasets for the tasks include:
ECS - (https://github.com/karoldvl/ESC-50) 
- 50 classes of noise categories
- 40 audio records 
- 5 seconds each

we ran this project with only 10 classes (that are in ECS-10). In this github we can see the top models performances. 

By setting ONLY_10_LABELS=False in models/audio_ecs/utils would run the model with 50 classes rather then 10


### Sport-images:

<br>
<br>

<!-- ![WhatsApp Image 2024-04-07 at 21 04 28_e74cb264](https://github.com/DanielLevi6/046211-Deep-Learning/assets/88712194/c96e8a3b-c3f8-4f27-8d41-56e13096ba48) -->

## Model
### ECS-50 - 

we used well known model for audio named Hubert - (https://arxiv.org/pdf/2106.07447)

this is a transformer based on Bert using CNN for the first layers and projection after the end layers.

Sport-images:


## Repository Structure
- **optimizers:** containing simple check we did with the optimizers
- **models:**  containing code for the two task - audio_ecs and imagenet
    - models/audio_ecs - folder for the audio task:
        - data_preperation - showing some analysis and details about the data
        - model - file to train the model
        - optimizers_optuna - file for optune optimizers hyperparameters for our specific task
        - utils - general functions used for the files above  



## How to Use
To utilize our earthquake prediction model, follow these steps:
1. **Clone the Repository:** Clone this repository to your local machine.
2. **Install Dependencies:** Ensure all necessary dependencies are installed.
3. **Prepare Data:** If using custom data, format it appropriately and replace the existing dataset.
4. **Run the Model:** Execute provided scripts in the `code` directory.
5. **Evaluate Results:** Examine predictions and evaluate model performance.

## Libraries Used
the Requirements.txt file in this repo can be used to download all of the required libraries.

## Results Summary

Audio: 

### first failure - madgrad without gradient clipping perform really bad:

![screenshot](models/audio_ecs/results/madgrad_failure.PNG)

### first Trail using hyperparameters:

![screenshot](models/audio_ecs/results/first_trail_train.PNG)
![screenshot](models/audio_ecs/results/first_trail_validation.PNG)

### Second Trail using hyperparameters:
![screenshot](models/audio_ecs/results/second_trail_train.PNG)
![screenshot](models/audio_ecs/results/second_trail_validation.PNG)

### Second Trail for more epochs:

![screenshot](models/audio_ecs/results/second_trail_500.PNG)



## Conclusion

### Audio
1. Gradient clipping​ for madgrad was crucial for optimizing hubert with it

2. SGD ​- under perform the other optimizers for this task

3. Train vs Validation – ​

    - train - Adan, Adam and scheduleFree​ - converge fast for this task

    - val - the optimizers reach a similar point 

4. Noise - Adam and scheduleFree were the most smooth - Madgrad and Adan were more noisy
for noisy optimizers we recommend for early stopping ​

5. Who is better? we didnt see a clear separation in the accuracy for the models– ​

    we can assume that as the articles says for different choice of optimizer 
    we may see better performance than Adam.
    so we recommend try using schedule free and Adan 


## Future Work

- **Training more for a simple task::**
    
    We would like to see a clear separation in the graphs for a specific task.  

    We couldn't match the amount of training for the 3 optimizers with our computation resources and maybe that is the reason why. 

    training for a simple task but for many epochs would help to see any seperattion as in the article

- **Check image classification with specific objects:**

## Contact
For inquiries or further information, please contact roy.weber@campus.technion.ac.il, tomer.rudich@campus.technion.ac.il
Thank you for your interest in our Optimizers project.