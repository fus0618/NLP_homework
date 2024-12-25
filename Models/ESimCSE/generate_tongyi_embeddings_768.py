import os
import json
import tempfile
from tqdm import tqdm
from loguru import logger
from typing import List, Union, Tuple
from openai import OpenAI

# --------------------- Read API Key and URL ---------------------

def read_api_key_and_url(file_path: str) -> Tuple[str, str]:
    """
    从文件中读取API Key和基础URL。

    文件格式：
    第一行：API Key
    第二行：基础URL
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            api_key = lines[0].strip()
            base_url = lines[1].strip()
            return api_key, base_url
    except FileNotFoundError:
        print("Error: API key file not found.")
        return None, None
    except IndexError:
        print("Error: File format is incorrect. Make sure the first line is the API key and the second line is the URL.")
        return None, None

# 从文件中读取API Key和URL
file_path = 'F:/tongyi_api_key.txt'  # 确保文件路径正确
api_key, base_url = read_api_key_and_url(file_path)
if not api_key or not base_url:
    exit(1)

# 确保 base_url 正确
if base_url.endswith('/'):
    base_url = base_url.rstrip('/')

# --------------------- Custom OpenAI Client ---------------------

class CustomOpenAIClient:
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化 OpenAI 兼容的客户端。
        :param api_key: 用户的 API Key。
        :param base_url: 服务的 base_url。
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_embeddings(
        self,
        input_data: Union[str, List[str]],
        model: str = "text-embedding-v3",
        dimensions: int = 768,  # 指定为768维嵌入
        encoding_format: str = "float"
    ) -> List[List[float]]:
        """
        调用百炼的 Embedding API 获取嵌入向量。
        :param input_data: 输入的文本，可以是字符串或字符串列表。
        :param model: 使用的模型名，默认为 `text-embedding-v3`。
        :param dimensions: 指定输出向量的维度。
        :param encoding_format: 控制返回的嵌入向量格式，默认为 `float`。
        :return: 嵌入向量的列表。
        """
        if isinstance(input_data, str):
            input_data = [input_data]  # 转为列表

        completion = self.client.embeddings.create(
            model=model,
            input=input_data,
            dimensions=dimensions,
            encoding_format=encoding_format
        )
        embeddings = [item.embedding for item in completion.data]
        return embeddings

# 初始化客户端
client = CustomOpenAIClient(api_key=api_key, base_url=base_url)

# --------------------- Generate Teacher Embeddings ---------------------

def generate_teacher_embeddings(sentences: List[str], save_path: str, client: CustomOpenAIClient, batch_size: int = 20):
    """
    遍历训练语料并生成教师模型嵌入，支持断点续传，保存在指定路径。
    """
    # 检查是否已经存在缓存
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                cached_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load existing cache from {save_path}. Starting fresh.")
                cached_data = {}
    else:
        cached_data = {}

    # 确认未处理的句子
    embedding_status = {sent: "embedding" in data and data["done"] for sent, data in cached_data.items()}
    remaining_sentences = [sent for sent in sentences if not embedding_status.get(sent, False)]
    logger.info(f"Total sentences: {len(sentences)}, Remaining to embed: {len(remaining_sentences)}")

    # 批量生成嵌入
    for i in tqdm(range(0, len(remaining_sentences), batch_size), desc="Generating embeddings"):
        batch = remaining_sentences[i:i + batch_size]
        embeddings = client.get_embeddings(batch)  # 调用 API

        # 保存结果
        for sent, emb in zip(batch, embeddings):
            if emb is not None:
                cached_data[sent] = {"embedding": emb, "done": True}
            else:
                cached_data[sent] = {"embedding": None, "done": False}
                logger.warning(f"Failed to generate embedding for: {sent}")

        # 临时文件保存进度
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=os.path.dirname(save_path)) as tmp_file:
                json.dump(cached_data, tmp_file, ensure_ascii=False, indent=2)
                temp_name = tmp_file.name
            os.replace(temp_name, save_path)  # 原子性保存
        except Exception as e:
            logger.error(f"Failed to save cache to {save_path}: {e}")
            raise e

    logger.info(f"Embedding generation completed. Saved to {save_path}.")

# --------------------- Main Execution ---------------------

def load_cnsd_sts_train_unsup(file_path: str) -> List[str]:
    """
    加载 `cnsd-sts-train_unsup.txt` 文件中的句子。
    :param file_path: 文件路径
    :return: 句子列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

if __name__ == "__main__":
    # 数据文件路径
    train_unsup_path = "../data/STS-B/cnsd-sts-train_unsup.txt"  # 请确保此路径正确

    # 保存嵌入的路径
    save_path = "./cnsd_sts_train_unsup_embeddings_768.json"

    # 加载数据
    sentences = load_cnsd_sts_train_unsup(train_unsup_path)
    logger.info(f"Loaded {len(sentences)} sentences from {train_unsup_path}.")

    # 生成嵌入
    generate_teacher_embeddings(sentences, save_path, client, batch_size=6)
    logger.info(f"Embeddings saved to {save_path}.")
