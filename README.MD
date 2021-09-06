# Raw Action vs. Human Action in AlphaStar

## Introduction

Official code for the paper "Rethinking of AlphaStar".

## Usage

To change whether use raw interface (raw actions) or human interface (human actions), please change in the param.py:

line 22: use_raw_action = False

* If it is True, meaning we are using raw action space.
* If it is False, meaning we are using human action space.

To run the training, just type "python run.py".

Other parameters are not needed to change.

## Requirements

The python package requirements for running the codes are presented in the "requirements.txt".