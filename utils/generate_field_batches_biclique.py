import json
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from fim import eclat
import os
import random
import logging
from collections import Counter

from utils.detect_column_types import detect_column_types
from utils.table import markdown_table_to_listdict

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_field_batches_biclique( 
        id2d,
        min_samples,
        max_samples,
        min_fields,
        max_fields,
        sport_min_samples,
        sport_max_samples,
        sport_min_fields,
        sport_max_fields,
        cache_file,
        max_num
    ):
    
    if not os.path.exists(cache_file):
        # 步骤 1: 预处理 - 遍历一次所有表格，提取所有必要信息
        logger.warning("未检测到缓存文件，开始预处理数据...")
        table_metadata = []
        fail_num = 0
        for task_id, data in id2d.items():
            if 'unique_columns' not in data or 'required' not in data or 'domain' not in data:
                continue
            
            answer_list = markdown_table_to_listdict(data['answer'])
            col_types = detect_column_types(answer_list)
            
            pk_cols = data['unique_columns']
            pk_col_types = [col_types[pk_col] for pk_col in pk_cols]
            pk_type_key = tuple(sorted(pk_col_types))

            transaction = tuple(
                (data["domain"], col, col_types[col])
                for col in data['required']
                if col in col_types # 确保列存在
            )

            if not transaction: # 如果没有有效的列，则跳过
                continue

            table_metadata.append({
                'task_id': task_id,
                'pk_count': len(pk_cols),
                'pk_type_key': pk_type_key,
                'transaction': transaction
            })
            
        
        logger.warning(f"预处理完成，成功处理 {len(table_metadata)} 个表格，失败 {fail_num} 个。")

        groups = defaultdict(lambda: defaultdict(list))
        for meta in table_metadata:
            groups[meta['pk_count']][meta['pk_type_key']].append(meta)

        logger.warning("运行Eclat并生成批次...")
        all_batches = []
        non_single_pk_matches_count = 0
        
        process_desc = "处理 (主键数 -> 主键类型) 分组"
        for pk_count, sub_groups in tqdm(groups.items(), desc=process_desc, dynamic_ncols=True):
            for pk_type_key, tables_in_group in sub_groups.items():
                
                if len(tables_in_group) < min_samples:
                    continue

                # 直接从元数据构建 transactions 和 field_to_task_ids，无需再次计算
                transactions = [meta['transaction'] for meta in tables_in_group]
                
                field_to_task_ids = defaultdict(list)
                for meta in tables_in_group:
                    for typed_field in meta['transaction']:
                        field_to_task_ids[typed_field].append(meta['task_id'])
                
                closed_itemsets_with_support = eclat(transactions, supp=-min_samples, target='c')
                
                for common_fields_tuple, _ in closed_itemsets_with_support:
                    if not common_fields_tuple: continue
                    
                    domain_name = common_fields_tuple[0][0]
                    if domain_name.lower() == 'sports':
                        cur_min_samples, cur_max_samples, cur_min_fields, cur_max_fields = sport_min_samples, sport_max_samples, sport_min_fields, sport_max_fields
                    else:
                        cur_min_samples, cur_max_samples, cur_min_fields, cur_max_fields = min_samples, max_samples, min_fields, max_fields

                    if not (cur_min_fields <= len(common_fields_tuple) <= cur_max_fields):
                        continue

                    supporting_task_ids = set(field_to_task_ids[common_fields_tuple[0]])
                    for typed_field in common_fields_tuple[1:]:
                        supporting_task_ids.intersection_update(field_to_task_ids[typed_field])
                    
                    biclique_task_ids = list(supporting_task_ids)
                    if len(biclique_task_ids) < cur_min_samples:
                        continue

                    if pk_count != 1:
                        non_single_pk_matches_count += 1
                        continue

                    for i in range(0, len(biclique_task_ids), cur_max_samples):
                        chunk_ids = biclique_task_ids[i:i + cur_max_samples]
                        if len(chunk_ids) >= cur_min_samples:
                            batch_data = [id2d[task_id] for task_id in chunk_ids]
                            common_field_names = [field[1] for field in common_fields_tuple]
                            domain_set = set([field[0] for field in common_fields_tuple])

                            if len(domain_set) == 1:
                                domain_name_for_batch = domain_set.pop()
                                
                                try:
                                    representative_data = batch_data[0]
                                    pk_col_names = representative_data['unique_columns']
                                    rep_answer_list = markdown_table_to_listdict(representative_data['answer'])
                                    rep_col_types = detect_column_types(rep_answer_list)
                                    common_fields_info = [f"'{name}' ({ctype})" for _, name, ctype in common_fields_tuple]
                                    pk_fields_info = [f"'{name}' ({rep_col_types.get(name, 'N/A')})" for name in pk_col_names]
                                    # log_message = (
                                    #     f"  - Domain: {domain_name_for_batch}\n"
                                    #     f"  - 批次大小: {len(chunk_ids)}\n"
                                    #     f"  - 通用字段 (name, type): {', '.join(common_fields_info)}\n"
                                    #     f"  - 主键字段 (name, type): {', '.join(pk_fields_info)}\n"
                                    #     f"-----------------------------------------"
                                    # )
                                    # logger.warning(log_message)
                                except Exception as e:
                                    pass

                                all_batches.append((common_field_names, batch_data, domain_name_for_batch))

        sports_batch = [b for b in all_batches if b[2].lower() == 'sports']
        other_batch = [b for b in all_batches if b[2].lower() != 'sports']
        sports_batch_sampled = random.sample(sports_batch, min(20000, len(sports_batch)))
        new_all_batches = sports_batch_sampled + other_batch
        random.shuffle(new_all_batches)

        if new_all_batches:
            domain_list = [b[2] for b in new_all_batches]
            domain_counts = Counter(domain_list)
            logger.warning("--- 各领域在批次中的分布情况 ---")
            total_batches = sum(domain_counts.values())
            for domain, count in sorted(domain_counts.items()):
                logger.warning(f"领域 '{domain}': {count} 个批次")
            logger.warning(f"总计: {total_batches} 个批次")
            logger.warning("------------------------------------")

        with open(cache_file, 'w') as f:
            for b in new_all_batches:
                f.write(json.dumps({'common_field_names': b[0], 'batch_data': b[1], 'domain_name': b[2]}) + '\n')
        logger.warning(f"已将批次缓存到文件 {cache_file}.")

    else:
        logger.warning(f"检测到缓存文件 {cache_file}，直接加载...")
        with open(cache_file) as f:
            loaded_data = [json.loads(l) for l in f.readlines()]
        
        new_all_batches = []
        for item in tqdm(loaded_data, desc="检查缓存的批次", dynamic_ncols=True):
            common_field_names = item['common_field_names']
            batch_data = item['batch_data']
            domain_name = item['domain_name']

            if not batch_data: continue
            
            # try:
            #     representative_data = batch_data[0]
            #     pk_col_names = representative_data['unique_columns']
            #     rep_answer_list = markdown_table_to_listdict(representative_data['answer'])
            #     rep_col_types = detect_column_types(rep_answer_list)
            #     common_fields_info = [f"'{name}' ({rep_col_types.get(name, 'N/A')})" for name in common_field_names]
            #     pk_fields_info = [f"'{name}' ({rep_col_types.get(name, 'N/A')})" for name in pk_col_names]
            #     # log_message = (
            #     #     f"  - Domain: {domain_name}\n"
            #     #     f"  - 批次大小: {len(batch_data)}\n"
            #     #     f"  - 通用字段 (name, type): {', '.join(common_fields_info)}\n"
            #     #     f"  - 主键字段 (name, type): {', '.join(pk_fields_info)}\n"
            #     #     f"---------------------------------------------"
            #     # )
            #     # logger.warning(log_message)
            # except Exception as e:
            #     pass

            new_all_batches.append((common_field_names, batch_data, domain_name))
        
        logger.warning(f"需要处理 {len(new_all_batches)} 个批次.")
    
    real_all_batches = [(fields, samples) for fields, samples, _ in new_all_batches]

    if max_num > 0:
        return real_all_batches[:max_num]
    return real_all_batches