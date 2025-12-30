from huggingface_hub import hf_hub_download
import tarfile

# 下载单个批次
batch_file = hf_hub_download(
    repo_id="MYJOKERML/chinese-dialogue-speech-dataset",
    filename="batch_002.tar.gz",
    repo_type="dataset"
)

# 解压
with tarfile.open(batch_file, "r:gz") as tar:
    tar.extractall("./data")
