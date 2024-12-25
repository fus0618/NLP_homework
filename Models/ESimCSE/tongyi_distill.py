import argparse
import sys
import json
import os
import tempfile
import time
from typing import List, Dict, Tuple, Union

import httpx  # Ensure httpx library is installed
from tqdm import tqdm
from loguru import logger
import http.client

import numpy as np
from scipy.stats import spearmanr
from transformers import BertTokenizer

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ESimCSE_dataloader import TrainDataset, TestDataset, CollateFunc, load_sts_data, load_sts_data_unsup
from ESimCSE_Model import ESimcseModel, MomentumEncoder, MultiNegativeRankingLoss
from openai import OpenAI
from sklearn.decomposition import PCA


def apply_pca(teacher_embeddings: torch.Tensor, n_components: int = 768) -> torch.Tensor:
    """
    使用PCA对教师嵌入进行降维。

    :param teacher_embeddings: 原始教师嵌入，形状为 (num_samples, original_dim)
    :param n_components: 降维后的维度，默认为768
    :return: 降维后的教师嵌入，形状为 (num_samples, n_components)
    """

    original_dim = teacher_embeddings.size(1)  # 获取输入维度
    print("original_dim:", original_dim)
    print("PCA: n_components:", n_components)

    # 如果目标维度与原始维度相同，直接返回输入张量
    if n_components == original_dim:
        logger.info(f"PCA skipped because n_components ({n_components}) matches the input dimension ({original_dim}).")
        return teacher_embeddings

    pca = PCA(n_components=n_components)
    embeddings_np = teacher_embeddings.cpu().numpy()  # 转换为numpy数组
    pca.fit(embeddings_np)  # 训练PCA
    reduced_embeddings = pca.transform(embeddings_np)  # 应用PCA降维
    return torch.tensor(reduced_embeddings, dtype=torch.float32).to(teacher_embeddings.device)  # 转回torch.Tensor



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
file_path = 'F:/tongyi_api_key.txt'
api_key, base_url = read_api_key_and_url(file_path)
if not api_key or not base_url:
    exit(1)

# 确保 base_url 正确
if base_url.endswith('/'):
    base_url = base_url.rstrip('/')

class CustomOpenAIClient:
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化 OpenAI 兼容的客户端。
        :param api_key: 用户的 API Key，如果未传递则从环境变量 `DASHSCOPE_API_KEY` 中读取。
        :param base_url: 服务的 base_url，默认为百炼服务的 URL。
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_embeddings(
        self,
        input_data: Union[str, List[str]],
        model: str = "text-embedding-v3",
        dimensions: int = 1024,
        encoding_format: str = "float"
    ) -> List[List[float]]:
        """
        调用百炼的 Embedding API 获取嵌入向量。
        :param input_data: 输入的文本，可以是字符串或字符串列表。
        :param model: 使用的模型名，默认为 `text-embedding-v3`。
        :param dimensions: 指定输出向量的维度，仅适用于 `text-embedding-v3`。
        :param encoding_format: 控制返回的嵌入向量格式，默认为 `float`。
        :return: 嵌入向量的列表。
        """
        if isinstance(input_data, str):
            input_data = [input_data]  # 将字符串转换为列表以统一处理

        # 确保输入数据符合限制
        if len(input_data) > 6:
            raise ValueError("输入的字符串列表最多支持6条。")
        if any(len(sentence) > 8192 for sentence in input_data):
            raise ValueError("字符串长度不能超过8192个 Token。")

        # 调用 API 获取嵌入向量
        completion = self.client.embeddings.create(
            model=model,
            input=input_data,
            dimensions=dimensions,
            encoding_format=encoding_format
        )

        # print(completion)
        # 提取嵌入向量
        embeddings = [item.embedding for item in completion.data]


        return embeddings


# 初始化客户端，使用text-embedding-v3模型，dimension=1024
client = CustomOpenAIClient(base_url=base_url, api_key=api_key)


def generate_teacher_embeddings(sentences: List[str], save_path: str, client: CustomOpenAIClient, batch_size: int = 20):
    """
    遍历训练语料并生成教师模型（text-embedding-v3）嵌入，支持断点续传。
    仅为未生成嵌入的句子调用API，不填充零向量。
    """
    # 检查是否已经存在缓存
    if os.path.exists(save_path):
        with open(save_path, "r", encoding='utf-8') as f:
            try:
                cached_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in file {save_path}: {str(e)}")
                logger.error("JSON decode error. Starting with empty cache.")
                cached_data = {}
    else:
        cached_data = {}

    # 创建嵌入状态字典，键为句子，值为True/False
    embedding_status = {sent: (True if 'embedding' in data else False) for sent, data in cached_data.items()}

    # 找出未处理的句子（没有嵌入或嵌入失败的句子）
    remaining_sentences = [sent for sent in sentences if not embedding_status.get(sent, False)]

    logger.info(f"Total sentences: {len(sentences)}, Remaining to embed: {len(remaining_sentences)}")

    # 批量生成嵌入
    for i in tqdm(range(0, len(remaining_sentences), batch_size), desc="Generating embeddings"):
        batch = remaining_sentences[i:i + batch_size]

        embeddings = client.get_embeddings(batch)  # 使用自定义通义API调用

        # 处理嵌入结果
        for sent, emb in zip(batch, embeddings):
            if emb is not None:
                cached_data[sent] = {"embedding": emb, "done": True}
            else:
                cached_data[sent] = {"embedding": None, "done": False}
                logger.warning(f"Embedding for sentence failed: {sent}")

        # 使用临时文件保存进度，确保写入过程完整
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=os.path.dirname(save_path)) as tmp_file:
                json.dump(cached_data, tmp_file, ensure_ascii=False, indent=2)
                temp_name = tmp_file.name
            os.replace(temp_name, save_path)  # 原子性替换
        except Exception as e:
            logger.error(f"Failed to write cache to {save_path}: {e}")
            raise e

    logger.info(f"Embedding generation completed. Saved to {save_path}")

def load_teacher_embeddings(load_path: str, sentences: List[str]) -> torch.Tensor:
    """
    加载本地缓存的文本嵌入。
    返回一个按提供句子顺序排列的torch.Tensor，仅包含已生成嵌入的句子。
    """
    with open(load_path, "r", encoding='utf-8') as f:
        cached_data = json.load(f)

    embeddings = []
    missing_sentences = []
    embedding_dim = 1536  # 根据模型调整

    for sent in sentences:
        data = cached_data.get(sent)
        if data and data.get("done") and data.get("embedding"):
            embeddings.append(data["embedding"])
        else:
            logger.warning(f"Missing embedding for sentence: {sent}")
            missing_sentences.append(sent)

    if missing_sentences:
        logger.error(f"{len(missing_sentences)} embeddings are missing. Please regenerate embeddings before training.")
        sys.exit(1)

    return torch.tensor(embeddings, dtype=torch.float32)


def distillation_loss(student_output: torch.Tensor, teacher_output: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    知识蒸馏损失函数，包括 L2 和 KL 散度。
    """
    # L2 损失
    l2_loss = torch.nn.functional.mse_loss(student_output, teacher_output)

    # KL 散度损失
    student_logits = student_output / temperature
    teacher_logits = teacher_output / temperature
    kl_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits, dim=-1),
        torch.nn.functional.softmax(teacher_logits, dim=-1),
        reduction="batchmean"
    )

    return l2_loss + kl_loss


# 线性层降维方法
# def train_with_distillation(model: ESimcseModel, momentum_encoder: MomentumEncoder, train_dl: DataLoader, dev_dl: DataLoader,
#                             optimizer: torch.optim.Optimizer, loss_func, teacher_embeddings: torch.Tensor,
#                             device: torch.device, save_path: str, gamma: float = 0.95):
#     """
#     结合文本嵌入的知识蒸馏训练。
#     """
#     model.train()
#     best = 0
#     teacher_embeddings = teacher_embeddings.to(device)  # 加载到 GPU/CPU
#
#     # 1. 定义投影层，将教师输出从1024降维到768
#     projection = nn.Linear(1024, 768).to(device)
#
#     # 2. 将投影层的参数添加到优化器中
#     optimizer.add_param_group({'params': projection.parameters()})
#
#     total_batches = len(train_dl)
#     for batch_idx, (batch_src_source, batch_pos_source, batch_neg_source) in enumerate(tqdm(train_dl, desc="Training"), start=1):
#         batch_size = batch_src_source['input_ids'].size(0)
#
#         # 获取学生模型输出
#         input_ids_src, attention_mask_src, token_type_ids_src = get_bert_input(batch_src_source, device)
#         src_out = model(input_ids_src, attention_mask_src, token_type_ids_src)  # shape: (batch_size, 768)
#
#         # 获取对应的教师嵌入
#         start_idx = (batch_idx - 1) * batch_size
#         end_idx = start_idx + batch_size
#         batch_teacher_output = teacher_embeddings[start_idx:end_idx].to(device)  # shape: (batch_size, 1024)
#
#         # 3. 将教师输出通过投影层降维到768
#         teacher_output_projected = projection(batch_teacher_output)  # shape: (batch_size, 768)
#
#         # # # 应用PCA降维
#         # logger.info("Applying PCA to teacher embeddings...")
#         # teacher_embeddings_pca = apply_pca(teacher_embeddings, n_components=768)
#
#         # 对比学习损失
#         input_ids_pos, attention_mask_pos, token_type_ids_pos = get_bert_input(batch_pos_source, device)
#         pos_out = model(input_ids_pos, attention_mask_pos, token_type_ids_pos)  # shape: (batch_size, 768)
#         contrastive_loss = loss_func(src_out, pos_out, None)
#
#         # 知识蒸馏损失（线性层）
#         teacher_loss = distillation_loss(src_out, teacher_output_projected)
#
#         # 知识蒸馏损失（PCA）
#         # teacher_loss = distillation_loss(src_out, teacher_embeddings_pca)
#         # 总损失
#         loss = contrastive_loss + 0.10 * teacher_loss  # 调整蒸馏权重
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # 更新 Momentum Encoder
#         for encoder_param, moco_encoder_param in zip(model.parameters(), momentum_encoder.parameters()):
#             moco_encoder_param.data = gamma * moco_encoder_param.data + (1. - gamma) * encoder_param.data
#
#         if batch_idx % 5 == 0 or batch_idx == total_batches:
#             logger.info(f'Batch {batch_idx}/{total_batches} - loss: {loss.item():.4f}')
#             corrcoef = evaluation(model, dev_dl, device)
#             model.train()
#             if best < corrcoef:
#                 best = corrcoef
#                 torch.save(model.state_dict(), save_path)
#                 logger.info(f"New best corrcoef: {best:.4f} at batch {batch_idx}, model saved.")

# PCA降维方法
def train_with_distillation(model: ESimcseModel, momentum_encoder: MomentumEncoder, train_dl: DataLoader, dev_dl: DataLoader,
                            optimizer: torch.optim.Optimizer, loss_func, teacher_embeddings: torch.Tensor,
                            device: torch.device, save_path: str, gamma: float = 0.95):
    """
    结合文本嵌入的知识蒸馏训练。
    """
    model.train()
    best = 0
    teacher_embeddings = teacher_embeddings.to(device)  # 加载到 GPU/CPU

    # 1. 应用 PCA 并将整个教师嵌入降维到 768
    logger.info("Applying PCA to teacher embeddings...")
    teacher_embeddings_pca = apply_pca(teacher_embeddings, n_components=768)  # shape: (10462, 768)

    total_batches = len(train_dl)
    for batch_idx, (batch_src_source, batch_pos_source, batch_neg_source) in enumerate(tqdm(train_dl, desc="Training"), start=1):
        batch_size = batch_src_source['input_ids'].size(0)

        # 获取学生模型输出
        input_ids_src, attention_mask_src, token_type_ids_src = get_bert_input(batch_src_source, device)
        src_out = model(input_ids_src, attention_mask_src, token_type_ids_src)  # shape: (batch_size, 768)

        # 获取对应的教师嵌入（按批次裁剪）
        start_idx = (batch_idx - 1) * batch_size
        end_idx = start_idx + batch_size
        batch_teacher_output = teacher_embeddings_pca[start_idx:end_idx].to(device)  # shape: (batch_size, 768)

        # 对比学习损失
        input_ids_pos, attention_mask_pos, token_type_ids_pos = get_bert_input(batch_pos_source, device)
        pos_out = model(input_ids_pos, attention_mask_pos, token_type_ids_pos)  # shape: (batch_size, 768)
        contrastive_loss = loss_func(src_out, pos_out, None)

        # 知识蒸馏损失
        teacher_loss = distillation_loss(src_out, batch_teacher_output)  # 当前批次的损失

        # 总损失
        loss = contrastive_loss + 0.05 * teacher_loss  # 调整蒸馏权重

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新 Momentum Encoder
        for encoder_param, moco_encoder_param in zip(model.parameters(), momentum_encoder.parameters()):
            moco_encoder_param.data = gamma * moco_encoder_param.data + (1. - gamma) * encoder_param.data

        if batch_idx % 5 == 0 or batch_idx == total_batches:
            logger.info(f'Batch {batch_idx}/{total_batches} - loss: {loss.item():.4f}')
            corrcoef = evaluation(model, dev_dl, device)
            model.train()
            if best < corrcoef:
                best = corrcoef
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best corrcoef: {best:.4f} at batch {batch_idx}, model saved.")


def evaluation(model: ESimcseModel, dataloader: DataLoader, device: torch.device) -> float:
    """
    模型评估，计算Spearman相关系数。
    """
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # Source input
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

            # Target input
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)

            # 计算相似度
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))

    # 计算Spearman相关系数
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation

def get_bert_input(source: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    获取BERT模型的输入。
    """
    input_ids = source.get('input_ids').to(device)
    attention_mask = source.get('attention_mask').to(device)
    token_type_ids = source.get('token_type_ids').to(device)
    return input_ids, attention_mask, token_type_ids

def main(args):

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {args.device} device.\n")

    # 加载数据路径
    train_path_unsup = os.path.join(args.data_path, "cnsd-sts-train_unsup.txt")
    test_path_sp = os.path.join(args.data_path, "cnsd-sts-test.txt")

    # 加载测试数据
    test_data_source = load_sts_data(test_path_sp)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    # 加载训练数据
    train_data_source = load_sts_data_unsup(train_path_unsup)
    train_sents = [data[0] for data in train_data_source]
    train_dataset = TrainDataset(train_sents)

    # 生成或更新文本嵌入
    logger.info("Generating/updating embeddings...")
    generate_teacher_embeddings(train_sents, args.teacher_save_path, client, batch_size=6)

    # 加载文本嵌入
    teacher_embeddings = load_teacher_embeddings(args.teacher_save_path, train_sents)

    # 创建DataLoader
    train_call_func = CollateFunc(tokenizer, max_len=args.max_length, q_size=args.q_size, dup_rate=args.dup_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                  collate_fn=train_call_func)

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"], "Invalid pooler type."


    model = ESimcseModel(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout,
                         off_dropout=True).to(args.device)

    # 初始化动量编码器
    momentum_encoder = MomentumEncoder(args.pretrain_model_path, args.pooler).to(args.device)

    # 初始化损失函数和优化器
    ESimCSELoss = MultiNegativeRankingLoss()
    esimcse_loss = ESimCSELoss.multi_negative_ranking_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    try:
        train_with_distillation(model, momentum_encoder,
                                train_dataloader, test_dataloader,
                                optimizer, esimcse_loss, teacher_embeddings,
                                args.device, args.save_path)
    except KeyboardInterrupt:
        logger.warning("训练过程中被中断。")
    except Exception as e:
        logger.error(f"训练过程中发生异常: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESimCSE with Tongyi Embedding Knowledge Distillation")


    parser.add_argument("--device", type=str, default='cuda:0', help="gpu or cpu")
    parser.add_argument("--save_path", type=str, default='./model_save', help="Path to save the best model")

    # # 保存模型嵌入（1024嵌入+降维）
    # parser.add_argument("--teacher_save_path", type=str, default='./tongyi_embeddings.json',
    #                     help="Path to save/load teacher embeddings")

    # 保存模型嵌入  直接使用768嵌入的json！
    parser.add_argument("--teacher_save_path", type=str, default='./cnsd_sts_train_unsup_embeddings_768.json',
                        help="Path to save/load teacher embeddings")

    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--dup_rate", type=float, default=0.15, help="Duplication rate for data augmentation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--q_size", type=int, default=64, help="Queue size for contrastive learning")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of input sentences")
    parser.add_argument("--data_path", type=str, default="../data/STS-B/", help="Path to the dataset")
    parser.add_argument("--pretrain_model_path", type=str, default=r"F:\models\bert-base-chinese",
                        help="Path to the pretrained BERT model")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='first-last-avg',
                        help='Which pooler to use')

    args = parser.parse_args()

    # 设置日志
    log_dir = "../log"
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "train.log"), level="DEBUG")  # 设置为DEBUG级别
    logger.info("Starting training process with knowledge distillation from Tongyi embeddings.")
    logger.info(args)

    # 测试单个请求
    test_sentences = ["这是一个测试句子。", "Another test sentence."]
    test_embeddings = client.get_embeddings(test_sentences)
    logger.info(f"Test Embeddings长度: {len(test_embeddings[0])}")
    # print(f"Test Embeddings: {test_embeddings}")

    main(args)
