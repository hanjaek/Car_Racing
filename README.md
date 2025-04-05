# 🚗 Evaluating Human-in-the-Loop Integration in RL-Based Autonomous Driving for Stability and Efficiency

## 📌 Overview
This project explores the integration of **Human-in-the-Loop (HIL)** techniques into **Reinforcement Learning (RL)** to improve the **stability** and **efficiency** of autonomous driving agents. We employ the **Soft Actor-Critic (SAC)** algorithm in the [CarRacing-v3](https://www.gymlibrary.dev/environments/box2d/car_racing/) environment using OpenAI's Gym.

## 🧠 Key Features
- Human intervention is injected during the early phase of training to guide the agent's behavior.
- Human actions are blended with the SAC policy using a decaying alpha coefficient.
- Human intervention data is stored in the same replay buffer as the agent's experience.
- After the intervention phase, the model continues to train autonomously using the collected data.

## 🧪 Experimental Focus
- Comparison between standard SAC and SAC+HIL models under identical seed settings.
- Evaluation of reward curves to analyze learning stability and final performance.
- Investigation into limitations when human data is not prioritized during training.

## 🛠️ Tech Stack
- Python
- [OpenAI Gym](https://github.com/openai/gym)
- [Stable-Baselines3 (SAC)](https://github.com/DLR-RM/stable-baselines3)
- Pygame (for keyboard-based human intervention)

## 🧪 Experimental Result (Preview)
![SAC Model Comparison](./graph/sac_comparison.png)  
*Figure 1. SAC model reward comparison*

![SAC+HIL Model Comparison](./graph/sac_hil_comparison.png)  
*Figure 2. SAC+HIL model reward comparison*


## 💡 Future Work
- Prioritized sampling of human intervention data
- Application of the method to different environments or RL algorithms
- Improving the quality and accuracy of human input for better guidance
