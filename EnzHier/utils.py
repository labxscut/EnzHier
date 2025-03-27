import os
import random
import numpy as np
import torch

def ensure_dirs(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
    - directory: 要确保存在的目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def seed_everything(seed=42):
    """
    设置随机种子以确保结果可重现
    
    参数:
    - seed: 随机种子值，默认为42
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")
    
import csv
import random
import os
import math
from re import L
import torch
import numpy as np
import subprocess
import pickle
from .distance_map import get_dist_map

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id


def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a


def load_esm(lookup):
    """
    加载ESM嵌入向量
    
    参数:
    - lookup: 要加载的蛋白质ID
    
    返回:
    - 加载的ESM嵌入向量，经过扩展维度
    """
    file_path = './data/esm_data/' + lookup + '.pt'
    
    # 首先检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ESM嵌入文件不存在: {file_path}，请先运行retrive_esm1b_embedding函数生成嵌入文件。")
    
    try:
        # 尝试加载文件
        embedding = torch.load(file_path)
        esm = format_esm(embedding)
        return esm.unsqueeze(0)
    except Exception as e:
        # 捕获所有可能的异常
        error_msg = f"加载ESM嵌入文件失败: {file_path}，错误: {str(e)}"
        # 提供更多详细信息
        if "PytorchStreamReader failed locating file data.pkl" in str(e):
            error_msg += "\n文件格式可能不兼容或已损坏。请重新生成ESM嵌入文件。"
        raise RuntimeError(error_msg)


def esm_embedding(ec_id_dict, device, dtype):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    failed_ids = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        valid_esm_to_cat = []
        
        for protein_id in ids_for_query:
            try:
                embedding = load_esm(protein_id)
                valid_esm_to_cat.append(embedding)
            except Exception as e:
                failed_ids.append(protein_id)
                print(f"警告: 无法加载蛋白质ID '{protein_id}'的ESM嵌入: {str(e)}")
                continue
        
        if valid_esm_to_cat:
            esm_emb = esm_emb + valid_esm_to_cat
    
    if failed_ids:
        print(f"警告: 有{len(failed_ids)}个蛋白质ID无法加载ESM嵌入文件")
        print(f"前5个失败的ID: {failed_ids[:5]}")
        
    if not esm_emb:
        raise RuntimeError("所有ESM嵌入文件都无法加载，请检查数据目录和文件格式")
    
    return torch.cat(esm_emb).to(device=device, dtype=dtype)


def model_embedding_test(id_ec_test, model, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    ids_for_query = list(id_ec_test.keys())
    valid_esm_to_cat = []
    valid_ids = []
    failed_ids = []
    
    for protein_id in ids_for_query:
        try:
            embedding = load_esm(protein_id)
            valid_esm_to_cat.append(embedding)
            valid_ids.append(protein_id)
        except Exception as e:
            failed_ids.append(protein_id)
            print(f"警告: 无法加载蛋白质ID '{protein_id}'的ESM嵌入: {str(e)}")
            continue
    
    if failed_ids:
        print(f"警告: 有{len(failed_ids)}个蛋白质ID无法加载ESM嵌入文件")
        print(f"前5个失败的ID: {failed_ids[:5]}")
        
    if not valid_esm_to_cat:
        raise RuntimeError("所有ESM嵌入文件都无法加载，请检查数据目录和文件格式")
    
    # 创建一个新的字典，仅包含有效的ID
    valid_id_ec_test = {id: id_ec_test[id] for id in valid_ids}
    
    esm_emb = torch.cat(valid_esm_to_cat).to(device=device, dtype=dtype)
    model_emb = model(esm_emb)
    
    return model_emb, valid_id_ec_test

def model_embedding_test_ensemble(id_ec_test, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    valid_esm_to_cat = []
    valid_ids = []
    failed_ids = []
    
    for protein_id in ids_for_query:
        try:
            embedding = load_esm(protein_id)
            valid_esm_to_cat.append(embedding)
            valid_ids.append(protein_id)
        except Exception as e:
            failed_ids.append(protein_id)
            print(f"警告: 无法加载蛋白质ID '{protein_id}'的ESM嵌入: {str(e)}")
            continue
    
    if failed_ids:
        print(f"警告: 有{len(failed_ids)}个蛋白质ID无法加载ESM嵌入文件")
        print(f"前5个失败的ID: {failed_ids[:5]}")
        
    if not valid_esm_to_cat:
        raise RuntimeError("所有ESM嵌入文件都无法加载，请检查数据目录和文件格式")
    
    # 创建一个新的字典，仅包含有效的ID
    valid_id_ec_test = {id: id_ec_test[id] for id in valid_ids}
    
    esm_emb = torch.cat(valid_esm_to_cat).to(device=device, dtype=dtype)
    
    return esm_emb, valid_id_ec_test

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")
        
def retrive_esm1b_embedding(fasta_name):
    """
    生成ESM-1b模型的蛋白质嵌入向量
    
    参数:
    - fasta_name: FASTA格式文件的名称（不含扩展名）
    """
    esm_script = "esm/scripts/extract.py"
    esm_out = "data/esm_data"
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_path = "data/" + fasta_name + ".fasta"
    
    # 确保输出目录存在
    ensure_dirs(esm_out)
    
    # 检查FASTA文件是否存在
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA文件不存在: {fasta_path}")
    
    print(f"开始生成{fasta_name}的ESM-1b嵌入...")
    
    try:
        command = ["python", esm_script, esm_type, 
                  fasta_path, esm_out, "--include", "mean"]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ESM-1b嵌入生成错误: {result.stderr}")
            raise RuntimeError(f"生成ESM-1b嵌入失败，返回代码: {result.returncode}")
        
        print(f"ESM-1b嵌入生成成功，保存在目录: {esm_out}")
    except Exception as e:
        print(f"生成ESM-1b嵌入时发生错误: {str(e)}")
        raise
    
def retrive_esm2b_embedding(fasta_name):
    """
    生成ESM-2b模型的蛋白质嵌入向量
    
    参数:
    - fasta_name: FASTA格式文件的名称（不含扩展名）
    """
    esm_script = "esm/scripts/extract.py"
    esm_out = "data/esm_data_1"
    esm_type = "esm2b_t36_3B_UR50D"
    fasta_path = "data/" + fasta_name + ".fasta"
    
    # 确保输出目录存在
    ensure_dirs(esm_out)
    
    # 检查FASTA文件是否存在
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA文件不存在: {fasta_path}")
    
    print(f"开始生成{fasta_name}的ESM-2b嵌入...")
    
    try:
        command = ["python", esm_script, esm_type, 
                  fasta_path, esm_out, "--include", "mean"]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ESM-2b嵌入生成错误: {result.stderr}")
            raise RuntimeError(f"生成ESM-2b嵌入失败，返回代码: {result.returncode}")
        
        print(f"ESM-2b嵌入生成成功，保存在目录: {esm_out}")
    except Exception as e:
        print(f"生成ESM-2b嵌入时发生错误: {str(e)}")
        raise
    
def retrive_esm2_embedding(fasta_name):
    """
    生成ESM-2模型的蛋白质嵌入向量
    
    参数:
    - fasta_name: FASTA格式文件的名称（不含扩展名）
    """
    esm_script = "esm/scripts/extract.py"
    esm_out = "data/esm_testset_20"
    esm_type = "esm2_t36_3B_UR50D" # embedding dim=1280
    fasta_path = "data/" + fasta_name + ".fasta"
    
    # 确保输出目录存在
    ensure_dirs(esm_out)
    
    # 检查FASTA文件是否存在
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA文件不存在: {fasta_path}")
    
    print(f"开始生成{fasta_name}的ESM-2嵌入...")
    
    try:
        command = ["python", esm_script, esm_type, 
                  fasta_path, esm_out, "--include", "mean"]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ESM-2嵌入生成错误: {result.stderr}")
            raise RuntimeError(f"生成ESM-2嵌入失败，返回代码: {result.returncode}")
        
        print(f"ESM-2嵌入生成成功，保存在目录: {esm_out}")
    except Exception as e:
        print(f"生成ESM-2嵌入时发生错误: {str(e)}")
        raise

def compute_esm_distance(train_file):
    ensure_dirs('./data/distance_map/')
    _, ec_id_dict = get_ec_id_dict('./data/' + train_file + '.csv')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    esm_emb = esm_embedding(ec_id_dict, device, dtype)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    pickle.dump(esm_dist, open('./data/distance_map/' + train_file + '.pkl', 'wb'))
    pickle.dump(esm_emb, open('./data/distance_map/' + train_file + '_esm.pkl', 'wb'))
    
def prepare_infer_fasta(fasta_name):
    """
    从FASTA文件准备推理所需的CSV文件，并生成ESM嵌入
    
    参数:
    - fasta_name: FASTA格式文件的名称（不含扩展名）
    """
    fasta_path = './data/' + fasta_name + '.fasta'
    csv_path = './data/' + fasta_name + '.csv'
    
    # 检查FASTA文件是否存在
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA文件不存在: {fasta_path}")
    
    print(f"开始处理FASTA文件: {fasta_path}")
    
    try:
        # 生成ESM嵌入
        print("正在生成ESM-1b嵌入...")
        retrive_esm1b_embedding(fasta_name)
        
        # 创建CSV文件
        print(f"正在创建CSV文件: {csv_path}")
        csvfile = open(csv_path, 'w', newline='')
        csvwriter = csv.writer(csvfile, delimiter = '\t')
        csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
        
        # 从FASTA文件提取序列标识符
        fastafile = open(fasta_path, 'r')
        entries_count = 0
        
        for line in fastafile.readlines():
            if line[0] == '>':
                entry_id = line.strip()[1:]
                csvwriter.writerow([entry_id, ' ', ' '])
                entries_count += 1
        
        csvfile.close()
        fastafile.close()
        
        print(f"处理完成。已提取{entries_count}个序列标识符到CSV文件。")
        print(f"ESM嵌入文件保存在'./data/esm_data/'目录中。")
        
    except Exception as e:
        print(f"准备推理FASTA时发生错误: {str(e)}")
        raise

def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id, csv_name, fasta_name) :
    csv_file = open('./data/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open('./data/' + fasta_name + '.fasta','w')
    single_id = set(single_id)
    for i, rows in enumerate(csvreader):
        if rows[0] in single_id:
            for j in range(10):
                seq = rows[2].strip()
                mu, sigma = .10, .02 # mean and standard deviation
                s = np.random.normal(mu, sigma, 1)
                mut_rate = s[0]
                times = math.ceil(len(seq) * mut_rate)
                for k in range(times):
                    position = random.randint(1 , len(seq) - 1)
                    seq = mutate(seq, position)
                seq = seq.replace('*', '<mask>')
                output_fasta.write('>' + rows[0] + '_' + str(j) + '\n')
                output_fasta.write(seq + '\n')

def mutate_single_seq_ECs(train_file):
    id_ec, ec_id =  get_ec_id_dict('./data/' + train_file + '.csv')
    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
    single_id = set()
    for id in id_ec.keys():
        for ec in id_ec[id]:
            if ec in single_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                single_id.add(id)
                break
    print("Number of EC numbers with only one sequences:",len(single_ec))
    print("Number of single-seq EC number sequences need to mutate: ",len(single_id))
    print("Number of single-seq EC numbers already mutated: ", len(single_ec) - len(single_id))
    mask_sequences(single_id, train_file, train_file+'_single_seq_ECs')
    fasta_name = train_file+'_single_seq_ECs'
    return fasta_name