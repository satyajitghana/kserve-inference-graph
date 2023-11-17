# KServe Inference Graph

Create VEnv

```bash
virtualenv venv --python=python3.10
```

```bash
source venv/bin/activate
```

Install requirements

```bash
pip install -r requirements-torch.txt
pip install -r requirements.txt
```

## MAR Generation

```bash
torch-model-archiver --model-name cat-classifier --handler ts_handlers/hf-image-classification/hf_image_classification_handler.py --requirements-file ts_handlers/hf-image-classification/requirements.txt --extra-files models/cat-classifier/ --version 1.0
```

## Test Model in TorchServe

```bash
torchserve --model-store model-store/cat-classifier/model-store --start --models all --foreground
```

## Docker Installation

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

## AWS CLI Setup


```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

```bash
aws s3 cp --recursive model-store s3://tsai-emlo/kserve-ig/
```

## Minikube

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

```bash
minikube start --driver=qemu --memory 40960 --cpus 16
```

```bash
minikube start --driver=docker --memory 12288 --cpus 4
```

For 5 models you'll need this, each model will take 1 vCPU and atleast 2GiB RAM each

```bash
minikube start --driver=docker --memory 28672 --cpus 8 --disk-size 180g
```

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

Exposing MiniKube to EC2 Public IP

```bash
minikube tunnel --bind-address 0.0.0.0
```


## KServe Installation

```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
```

## Notes

### JAVA Installation

```bash
sudo apt install default-jdk
```

```bash
update-alternatives --config java
```

Put `JAVA_HOME="/lib/jvm/java-11-openjdk-amd64"` in `.bashrc`


```
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./a.json
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" "http://127.0.0.1:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./a.json
```
