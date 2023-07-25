# Pretrain

Pretrain is a Python package that allows anyone in the world add computing power to a language model training job and get paid for it.

## Prerequisites

Pretrain requires Python 3.8 or above. You can check your Python version using the following command:

```bash
python --version
```

If your Python version is below 3.8, consider using a tool like [pyenv](https://github.com/pyenv/pyenv) to manage multiple Python versions on your machine.
Or install python via [Linux Install](https://iohk.zendesk.com/hc/en-us/articles/16724475448473-Install-Python-3-11-on-ubuntu) or [Mac Install](https://pythontest.com/python/installing-python-3-11/)

## Installation

First, clone the repository from GitHub:

```bash
git clone https://github.com/unconst/pretrain.git
```

Next, navigate to the cloned directory:

```bash
cd pretrain
```

Then, install the package:

```bash
pip install .
```

Note: It's a good practice to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to avoid conflicts with other Python projects.

Optionaly install [Weights and Biases](https://docs.wandb.ai/quickstart) and login via:
```bash
wandb login
```

## Usage

You can run the pretrain package using the following command:

```bash
python run.py <arguments>
```

To see a list of command-line arguments, run:

```bash
python pretrain/neuron.py --help
```

## Running with Docker

If you prefer to run pretrain in a Docker container, follow these instructions:

First, make sure Docker is installed on your machine. If it is not, you can download Docker [here](https://www.docker.com/products/docker-desktop).

Once Docker is installed and running, navigate to the pretrain directory and build the Docker image. Replace `your_image_name` with a name of your choice:

```bash
docker build -t your_image_name .
```

This command builds a Docker image from the Dockerfile and names it according to your preference.

After the image has been built, you can run the Docker container:

```bash
docker run your_image_name
```

This command starts a new Docker container and runs pretrain inside it.

Keep in mind that Docker containers are isolated from the host machine by default. If you need to share files between your host machine and the Docker container, or if you need to expose any ports, you'll need to specify this when you run the container. Refer to the Docker documentation for more details.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

---

