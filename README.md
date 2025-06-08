# PiCloud

## Overview

PiCloud is a web application that runs on a Raspberry Pi. It allows you to read from your notes and ask questions about them.

## Technologies

- [Streamlit](https://streamlit.io/)
- [Raspberry Pi](https://www.raspberrypi.org/)
- [Python](https://www.python.org/)

## System Setup

### Install Python
- Install pyenv
- pyenv install 3.13.1
- pyenv virtualenv 3.13.1 venv
- pyenv activate venv

### Install Ollama
- Install Ollama
- ollama run gemma3:1b 

## Setup

1. Clone the repository

```bash
git clone https://github.com/yourusername/PiCloud.git
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
streamlit run app.py
```



