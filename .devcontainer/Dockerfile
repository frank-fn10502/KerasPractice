FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

RUN apt update -y && \
    apt install -y \
    git \
    systemd \
    && pip3 install --upgrade pip 

# 設定時區
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends tzdata
RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata 

RUN pip install matplotlib && \
    pip install autopep8 && \
    
    #for docker
    pip install opencv-python-headless 
    

# 為了畫 MODEL 架構圖
RUN pip install pydot && \
    apt install -y graphviz 
    
#https://github.com/tensorflow/addons
RUN pip install tensorflow-addons==0.14.0 &&\

    pip install pyyaml && \
    pip install scipy