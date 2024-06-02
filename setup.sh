conda create -n MPR  python=3.9.0 -y
conda activate MPR
conda install cuda -c nvidia/label/cuda-11.8.0 # for compiling autoGPTQ
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.2;7.5;8.0;8.6+PTX;8.9;9.0"
GITHUB_ACTIONS=true
pip3 install -r requirements.txt