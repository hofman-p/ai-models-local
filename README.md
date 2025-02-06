# Introduction

What I wanted to check here:

- The possibility of running a model locally with Pytorch, Hugging Face and CUDA
- The possibility of fine-tuning, pruning and quantization of the model before exporting it
- The possibility to exporting the model to a mobile application (React Native) with Tensorflow Lite. I know it's extremely heavy to run AI on old phones and I wanted to challenge myself about it.

# How to run

Just a quick reminder for me for the things I did to make it work along the way (and for you who visits this and wanna test):

- Install CUDA: https://developer.nvidia.com/cuda-downloads
- Install last python version: https://www.python.org/downloads/
- Install Pytorch with CUDA support: https://pytorch.org/get-started/locally/
- Create an account on Hugging Face and get your API key (select the read to all models): https://huggingface.co/settings/tokens
- Require access to the models you'd like to test like this one: https://huggingface.co/mistralai/Mistral-7B-v0.1
- Import the model in the script (I tested one by one but you can update the script to test multiple models)
- Put your API key in the `.env` file (rename `.env.example` to `.env`)
- Run `py download_save_model.py`
- Run `py run_model.py`

# WIP

- Check for Tensorflow Lite
- Check for the linking with React Native
- Check for the good work of fine-tuning, pruning and quantization of the model before exporting it

# Reminder

- I used Windows 11 and I own a RTX 4090, so run these scripts at your own risk if you have a lower GPU
- Depending on which models you wanna test, you might need a lot of space on your disk
- If you wanna run it with on Mac, try CPU version of Pytorch and remove the CUDA part, also apparently BitsAndBytes is currently in WIP state for Mac so you might need another solution
