**Project Title**:
# Machine-Learning---Connect-4

**Project Description**:
This repository contains files to train two artificial agents to play connect 4 against each other.

**Project Goals**:

## Instructions for Build and Use

Steps to build and/or run the software:

1. Install the dependencies for the classic pettingzoo environment via the command: pip install 'pettingzoo[classic]'
2. If you run into the same error I did, ERROR: Failed building wheel for open-spiel, then run the command: pip install open-spiel
3. Update all python packages via the command: pip list --format freeze | %{pip install --upgrade $_.split('==')[0]}
4. If that doesn't fix it, use the command: pip install pettingzoo
6. Download and install the files in this reporitory
7. Add any dependencies that the code says are missing when you try to run it

Instructions for using the software:

1. Open the chat_gpt_connect_4.py file in Visual Studio Code and hit run.

## Development Environment 

To recreate the development environment, you need the following software and/or libraries with the specified versions:

* Python version 3.11.2 - 64 bit
* Visual Studio Code version 1.96.4
* gymnasium version 1.0.0
* matplotlib version 3.10.0
* numpy version 2.2.3
* pettingzoo version 1.24.3
* torch version 2.6.0
