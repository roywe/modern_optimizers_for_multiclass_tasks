# 046211_modern_optimizers_for_multiclass_tasks

# 046211 modern optimizers for multiclass tasks
Welcome to the to our project repository.
This repository houses the code and resources for checking new optimizers performances on one nlp and one vision task.
Our main goal is to check the optimizers validating their articles.


## Overview
In the recent years, new optimization methods arise.
while Adam is well known and used, different optimization technique claims that they are superior for specific cases.
Adan and Madgrad are two of those techniques. 
Adan - (https://arxiv.org/pdf/2208.06677) using nestrov momuntum show Sota results for vision transformer, bert and gpt and promsing results for RL method.
further download and documentation is here: (https://github.com/lucidrains/Adan-pytorch/tree/main)
Madgrad - (https://arxiv.org/pdf/2101.11075) - ................. 
further download and documentation is here: (https://github.com/facebookresearch/madgrad)


## Methodology
In this project, we tested the optimizers credability for NLP task and vision task.
We tested SGD, Adam, Madgrad and Adan with the 2 tasks and checked if loss and accuracy is behave as the article suggest.
Adan claims to be faster to converge vs other optimizors
Madgrad ......
We chose the following tasks:
-  ECS-50 Audio Dataset for multiclass NLP classification using Hubert
-  image............................ 


## Data
The datasets for the tasks include:
ECS - (https://github.com/karoldvl/ESC-50) containing 50 classes of 40 audio records of 5 seconds each
for this project we used only 10 classes (that are in ECS-10). In this github we can see the top models performances. 
Our goal was to compare different optimizers and not becoming SOTA for this task/

Images - 

<br>
<br>

<!-- ![WhatsApp Image 2024-04-07 at 21 04 28_e74cb264](https://github.com/DanielLevi6/046211-Deep-Learning/assets/88712194/c96e8a3b-c3f8-4f27-8d41-56e13096ba48) -->

## Model
ECS - we used well known model for audio named Hubert - (https://arxiv.org/pdf/2106.07447)
this is a Bert similar model using CNN for the first layers and projection after the end layers

Images - .................


## Repository Structure
- **optimizers:** containing simple check we did with the optimizers
- **models:**  containing code for the two task - audio_ecs and imagenet
- 


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


## Conclusion



## References



## Future Work


## Contact
For inquiries or further information, please contact roy.weber@campus.technion.ac.il, tomer.rudich@campus.technion.ac.il
Thank you for your interest in our Optimizers project.