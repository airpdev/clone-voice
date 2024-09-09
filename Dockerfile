FROM nvidia/cuda:11.8.0-base-ubuntu22.04
WORKDIR /app
COPY . .
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update -y
RUN apt install python3 python3-pip -y
RUN apt install -y ffmpeg
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install "fastapi[all]"
RUN pip3 install slowapi==0.1.8
RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install asyncio==3.4.3
RUN pip3 install requests==2.25.1
CMD [ "uvicorn", "test_clean:app", "--host", "0.0.0.0", "--port", "8080"]
EXPOSE 8080