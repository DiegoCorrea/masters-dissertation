# Dissertação de Mestrado
Este repositório possui o código da minha dissertação de mestrado realizada na Universidade Federal da Bahia no Programa de Pós-Graduação em Ciência da Computação (PGCOMP-UFBA). A pesquisa foi realizada durante o periodo de Fevereiro de 2020 até Julho de 2021, contando com o apoio financeiro da CAPES (88887.502736/2020-00) e apoio de maquinario dos Núcleo de Biologia Computacional e Gestão de Informações Biotecnológicas - NBCGIB e o Laboratório Nacional de Computação Científica (LNCC/MCTI, Brazil).  
O texto da dissertação pode ser encontrado no Repositório Institucional da UFBA sob o título []().
A pesquisa foi dividida em duas parte que são encontradas nos dois artigos:  
- Exploiting Personalized Calibration and Metrics for Fairness Recommendation
  - [Codigo](https://github.com/DiegoCorrea/Exploiting-Personalized-Calibration-and-Metrics-for-Fairness-Recommendation)  
  - [Docker Container no Code Ocean](https://doi.org/10.24433/CO.6790880.v1)  
  - [Artigo publicado na revista Expert Systems With Application](https://doi.org/10.1016/j.eswa.2021.115112)
- Em publicação...

Dúvidas e questionamentos devem ser enviados para o e-mail contido nos artigos.  

# Resumo da dissertação  
Sistemas de recomendação são ferramentas utilizadas para sugerir itens, que possivelmente sejam de interesse dos usuários. Estes sistemas baseiam-se no histórico de preferências do usuário para gerar uma lista de sugestões que possuam maior similaridade com o perfil do usuário, visando uma melhor precisão e um menor erro. É esperado que, ao ser recomendado um item, o usuário informe sua preferência ao sistema, indicando se gostou ou o quanto gostou do item recomendado. A interação do usuário com o sistema possibilita um melhor entendimento de seus gostos, que com o tempo, adiciona mais e mais itens a seu perfil de preferências. A recomendação baseada em similaridade do item com as preferências buscando a melhor precisão pode causar efeitos colaterais na lista como: superespecialização das recomendações em um determinado núcleo de itens, pouca diversidade de categorias e desbalanceamento de categoria ou gênero. Assim, esta dissertação tem como objetivo explorar a calibragem, que é um meio para produzir recomendações que sejam relevantes aos usuários e ao mesmo tempo considerar todas as áreas de suas preferências, buscando evitar a desproporção na lista de recomendação. Para isto, foram abordadas formas de ponderar o balanceamento entre a relevância das recomendações e a calibragem baseada em medidas de divergência, assim como um modelo de sistema calibrado e um protocolo de decisão. A hipótese é que a calibragem pode contribuir positivamente para recomendações mais justas de acordo com a preferência do usuário. A pesquisa foi realizada através de uma ampla abordagem propondo um modelo de sistema e um protocolo de decisão que contempla em seu experimento nove algoritmos de recomendação aplicados nos domínios de filme e música, analisando três medidas de divergência, dois pesos de balanceamento personalizado e dois balanceamentos entre relevância-calibragem. A avaliação foi analisada com métricas amplamente utilizadas, assim como métricas propostas neste trabalho. Os resultados indicam que a calibragem produz efeitos positivos tanto para a precisão da recomendação quanto para a justiça com as preferências do usuário, criando listas de recomendação que respeitem todas as áreas. Os resultados também indicam qual é a melhor combinação para obter um melhor desempenho ao aplicar as propostas de calibragem.

# OBSERVAÇÃO  
OBS 1: Tutorial de instalação com algumas lacunas. Verifique os nomes e diretorios.  
OBS 2: Caso use alguma parte deste código ou do artigo. Requisito que faça a devida citação.  

# To Ubuntu/Debian  

### Intall
1. Update and upgrade the OS: `sudo apt update && sudo apt upgrade -y`  
1. S.O. Installations: `sudo apt install -y git unzip htop gcc g++ gettext curl`  
1. [Conda Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)  
1.1. Download: `curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`   
1.1. Install: `bash Anaconda3-2020.11-Linux-x86_64.sh`  
1. PyCharm (Optional: for better development)  
  
## Config  
1. Git:  
.1. Reload bash: `source ~/.bashrc`  
.2. Generate ssh key: `ssh-keygen`  
.3. Copy the key: `cat ~/.ssh/id_rsa.pub`    
.4. Paste in the github  
.5. Clone the repository: `git clone git@github.com:DiegoCorrea/calibrated_recommendation.git`    
1. Go to the project root path: `cd calibrated_recommendation/`  
1. Load the conda environment: `conda env create -f environment.yml`  
1. Active the environment: `conda activate calibrated_recommendation`  

## Get Dataset
1. Get the dataset: `python get_dataset.py`    
2. Unzip the dataset: `unzip dataset.zip`  
3. Move to dataset dir  
  
## Run  
1. Extract language: `sh extract_language.sh`
1. Code on background: `python start_recommenders.py > log/output_terminal.log 2>&1 & disown`

# To RedHat/CentOS  

## Intall
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
  
## Config  
1. Git:   
.1. Generate ssh key: `ssh-keygen`  
.2. Copy the key: `cat ~/.ssh/id_rsa.pub`    
.3. Paste in the git(lab-hub)  
.4. Clone the repository: `git clone git@github.com:DiegoCorrea/calibrated_recommendation.git`    
1. Go to the project root path: `cd calibrated_recommendation/`  
1. Load the conda environment: `conda env create -f environment.yml`  
1. Active the environment: `conda activate calibrated_recommendation`  

## Get Dataset
Get the dataset: `python get_dataset.py`  
Unzip and move to dataset dir.  
  
## Run  
1. Extract language: `sh extract_language.sh`
1. Code on background: `python start_recommenders.py > logs/output_terminal.log 2>&1 & disown`    
