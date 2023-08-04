FROM nvidia/cuda:11.8.0-base-ubuntu22.04
WORKDIR /app
COPY . .
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update -y
RUN apt install python3 python3-pip -y
RUN apt install -y ffmpeg
RUN apt install -y git
RUN pip3 install boto3
RUN pip3 install lightning==2.0.1
RUN pip3 install funcy
RUN chmod +x ./run.sh
RUN ./run.sh
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install typing-extensions
RUN pip3 install .
RUN pip3 install "fastapi[all]"
RUN pip3 install uvicorn
RUN pip3 install joblib
RUN pip3 install fairseq
RUN pip3 install audiolm-pytorch
RUN pip3 install tensorboardX
RUN pip3 install ffmpeg-python
RUN pip3 install slowapi
RUN pip3 install soundfile
CMD [ "uvicorn", "generate_voice:app", "--host", "0.0.0.0", "--port", "8080"]
EXPOSE 8080
