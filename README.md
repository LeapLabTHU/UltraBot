# Towards Expert-level Autonomous Carotid Ultrasonography with Large-scale Learning-based Robotic System

<p align="center"> <img src='docs/intro.png' align="center" height="750px"> </p>

This repository is the official Pytorch implementation for paper **Towards Expert-level Autonomous Carotid Ultrasonography with Large-scale Learning-based Robotic System**. (Primary Contact: [Haojun Jiang](https://github.com/jianghaojun))

## Abstract

Carotid ultrasound requires skilled operators due to small vessel dimensions and high anatomical variability, exacerbating sonographer shortages and diagnostic inconsistencies. Prior automation attempts, including rule-based approaches with manual heuristics and reinforcement learning trained in simulated environments, demonstrate limited generalizability and fail to complete real-world clinical workflows. Here, we present UltraBot, a fully learning-based autonomous carotid ultrasound robot, achieving human-expert-level performance through four innovations: 
**(1)** A unified imitation learning framework for acquiring anatomical knowledge and scanning operational skills;
**(2)** A large-scale expert demonstration dataset (247,000 samples, 100 times scale-up), enabling embodied foundation models with strong generalization;
**(3)** A comprehensive scanning protocol ensuring full anatomical coverage for biometric measurement and plaque screening; 
**(4)** The clinical-oriented validation showing over 90\% success rates, expert-level accuracy, up to 5.5× higher reproducibility across diverse unseen populations.
Overall, we show that large-scale deep learning offers a promising pathway toward autonomous, high-precision ultrasonography in clinical practice.

## Demo: autonomous scanning, biometric measurement, and plaque screening.

For more demos, please refer to Supplementary Videos 1-3 in the article.

<p align="center"> <img src='docs/demo.gif' align="center" height="800px"> </p>


## Usage

This project consists of three main components:

- **action_decision**: Responsible for autonomous robotic decision-making during carotid ultrasound scanning.
- **biometric_measurement**: Handles anatomical landmark detection and automatic measurement of carotid intima-media thickness and lumen diameter.
- **plaque_segmentation**: Focuses on identifying and segmenting carotid plaques from ultrasound images.

Each module is organized in its respective subdirectory and includes a dedicated `README.md` file with detailed instructions on setup, training, and inference. Please refer to the `README.md` inside each folder for component-specific usage and guidelines.

## Contacts
jhj20 at mails dot tsinghua dot edu dot cn

Any discussions or concerns are welcomed!
