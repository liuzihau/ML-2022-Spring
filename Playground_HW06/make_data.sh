curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/LeoFeng/MLHW_6
unzip ./MLHW_6/faces.zip -d .