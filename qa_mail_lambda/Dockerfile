# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.8 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

# Load the BERT model from Huggingface and store it in the model directory
RUN mkdir model
RUN curl -L https://huggingface.co/its5Q/rugpt3large_mailqa/main/pytorch_model.bin -o ./model/pytorch_model.bin
RUN curl https://huggingface.co/its5Q/rugpt3large_mailqa/main/config.json -o ./model/config.json
RUN curl https://huggingface.co/its5Q/rugpt3large_mailqa/main/merges.txt -o ./model/merges.txt
RUN curl https://huggingface.co/its5Q/rugpt3large_mailqa/main/vocab.json -o ./model/vocab.json

# Set the CMD to your handler
CMD ["app.lambda_handler"]