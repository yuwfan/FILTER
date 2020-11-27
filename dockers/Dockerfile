FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt update && \
    apt install -y bash \
                   wget \
                   ssh \
                   vim \
                   build-essential \
                   git \
                   curl \
                   zip \
                   unzip \
                   ca-certificates \
                   libopenblas-dev \
                   libomp-dev \
                   autoconf \
                   automake \
                   libtool \
                   locales \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
RUN PWD_DIR=$(pwd)
RUN cd $(mktemp -d)
RUN git clone -q https://github.com/NVIDIA/apex.git
RUN cd apex; git reset --hard de6378f5dae8fcf2879a4be8ecea8bbcb9e59d5; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#RUN cd apex; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#  --global-option="--fast_multihead_attn" ./

RUN cd $(mktemp -d)
RUN git clone https://github.com/neubig/kytea.git
RUN cd kytea && autoreconf -i && ./configure && make && make install
RUN pip install kytea

RUN pip install tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk seqeval sacremoses pythainlp jieba faiss urllib3==1.25.4 networkx==1.11

#RUN cd $(mktemp -d)
#RUN git clone https://github.com/pytorch/fairseq
#RUN cd fairseq; pip install --editable ./
#RUN chmod -R 777 /opt/conda

RUN cd $PWD_DIR
RUN rm -rf /var/lib/apt/lists/* && locale-gen "en_US.UTF-8"
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

CMD ["/bin/bash"]
