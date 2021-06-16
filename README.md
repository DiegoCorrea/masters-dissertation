# Calibrated Recommendation  
Some introdutory text

## To Ubuntu/Debian  

### Intall
1. Update and upgrade the OS: `sudo apt update && sudo apt upgrade -y`  
1. S.O. Installations: `sudo apt install -y git unzip htop gcc g++ gettext curl`  
1. [Conda Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)  
1.1. Download: `curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`   
1.1. Install: `bash Anaconda3-2020.11-Linux-x86_64.sh`  
1. PyCharm (Optional: for better development)  
  
### Config  
1. Git:  
.1. Reload bash: `source ~/.bashrc`  
.2. Generate ssh key: `ssh-keygen`  
.3. Copy the key: `cat ~/.ssh/id_rsa.pub`    
.4. Paste in the github  
.5. Clone the repository: `git clone git@github.com:DiegoCorrea/calibrated_recommendation.git`    
1. Go to the project root path: `cd calibrated_recommendation/`  
1. Load the conda environment: `conda env create -f environment.yml`  
1. Active the environment: `conda activate calibrated_recommendation`  

### Get Dataset
1. Get the dataset: `python get_dataset.py`    
2. Unzip the dataset: `unzip dataset.zip`  
3. Move to dataset dir  
  
### Run  
1. Extract language: `sh extract_language.sh`
1. Code on background: `python start_recommenders.py > log/output_terminal.log 2>&1 & disown`

## To RedHat/CentOS  

### Intall
1. Update and upgrade the OS: `sudo yum update -y`  
1. Git: `sudo yum install git -y`  
1. Softwares de apoio: `sudo yum install openssl-devel bzip2-devel libffi-devel -y` 
1. Unzip: `sudo yum install unzip -y`  
1. htop: `sudo yum install htop -y`  
1. gcc e g++: `sudo yum install gcc gcc-c++ -y`
1. text: `sudo yum install gettext -y`
1. Python:  
.1. Download Python: `wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz`  
.2. Descompactar e ir para a pasta: `tar xzf Python-3.8.1.tgz && cd Python-3.8.1`  
.3. Configurar: `sudo ./configure --enable-optimizations`    
.4. Instalar: `sudo make altinstall`  
.5. Remove: `cd .. && sudo rm Python-3.8.1.tgz && sudo rm -rf Python-3.8.1`  
1. Conda [Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)-[All O.S.](https://docs.anaconda.com/anaconda/install/linux/):  
.1. Pre-instalação: `sudo yum install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver`    
.2. Download: `curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`   
.3. Install: `bash Anaconda3-2020.11-Linux-x86_64.sh -yes`  
.4. Reload bash: `source ~/.bashrc` 
1. PyCharm (Optional: for better development)  
  
### Config  
1. Git:   
.1. Generate ssh key: `ssh-keygen`  
.2. Copy the key: `cat ~/.ssh/id_rsa.pub`    
.3. Paste in the git(lab-hub)  
.4. Clone the repository: `git clone git@github.com:DiegoCorrea/calibrated_recommendation.git`    
1. Go to the project root path: `cd calibrated_recommendation/`  
1. Load the conda environment: `conda env create -f environment.yml`  
1. Active the environment: `conda activate calibrated_recommendation`  

### Get Dataset
Get the dataset: `python get_dataset.py`  
Unzip and move to dataset dir.  
  
### Run  
1. Extract language: `sh extract_language.sh`
1. Code on background: `python start_recommenders.py > logs/output_terminal.log 2>&1 & disown`    