# Use the mambaforge base image from condaforge
FROM condaforge/mambaforge:latest

# Install system dependencies
RUN conda install -c conda-forge -y gxx_linux-64 python=3.10

# Install PyTorch, torchvision, and torchaudio from the specified index URL
RUN pip install torch torchvision torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# Add the current directory to the container
ADD . /app
WORKDIR /app

# Install the Python dependencies from 'requirements.txt'
RUN pip install -r requirements.txt --no-cache-dir

# Install custom Pypose
RUN cd pypose && python setup.py develop

# The command to run when the container starts (modify this as required, it could be a service or a script)
CMD ["/bin/bash"]