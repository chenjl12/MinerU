# Copyright (c) Opendatalab. All rights reserved.
from loguru import logger
from openai import OpenAI
import json_repair

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text


def llm_aided_title(page_info_list, title_aided_config):
    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )
    title_dict = {}
    origin_title_list = []
    i = 0
    for page_info in page_info_list:
        blocks = page_info["para_blocks"]
        for block in blocks:
            if block["type"] == "title":
                origin_title_list.append(block)
                title_text = merge_para_with_text(block)

                if 'line_avg_height' in block:
                    line_avg_height = block['line_avg_height']
                else:
                    title_block_line_height_list = []
                    for line in block['lines']:
                        bbox = line['bbox']
                        title_block_line_height_list.append(int(bbox[3] - bbox[1]))
                    if len(title_block_line_height_list) > 0:
                        line_avg_height = sum(title_block_line_height_list) / len(title_block_line_height_list)
                    else:
                        line_avg_height = int(block['bbox'][3] - block['bbox'][1])

                title_dict[f"{i}"] = [title_text, line_avg_height, int(page_info['page_idx']) + 1]
                i += 1
    # logger.info(f"Title list: {title_dict}")

#     title_optimize_prompt = f"""输入的内容是一篇文档中所有标题组成的字典，请根据以下指南优化标题的结果，使结果符合正常文档的层次结构：

# 1. 字典中每个value均为一个list，包含以下元素：
#     - 标题文本
#     - 文本行高是标题所在块的平均行高
#     - 标题所在的页码

# 2. 保留原始内容：
#     - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
#     - 请务必保证输出的字典中元素的数量和输入的数量一致

# 3. 保持字典内key-value的对应关系不变

# 4. 优化层次结构：
#     - 根据标题内容的语义为每个标题元素添加适当的层次结构
#     - 行高较大的标题一般是更高级别的标题
#     - 标题从前至后的层级必须是连续的，不能跳过层级
#     - 标题层级最多为4级，不要添加过多的层级
#     - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息
#     - 字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们

# 5. 合理性检查与微调：
#     - 在完成初步分级后，仔细检查分级结果的合理性
#     - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
#     - 确保最终的分级结果符合文档的实际结构和逻辑

# IMPORTANT: 
# 请直接返回优化过的由标题层级组成的字典，格式为{{标题id:标题层级}}，如下：
# {{
#   0:1,
#   1:2,
#   2:2,
#   3:3
# }}
# 不需要对字典格式化，不需要返回任何其他信息。

# Input title list:
# {title_dict}

# Corrected title list:
# """
    #5.
    #- 字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们
    title_optimize_prompt = f"""
    你是一个文档标题结构分析助手。

    输入内容是一个字典，key 为标题 id（字符串），value 为：
    [
    标题文本,
    标题所在文本块的平均行高,
    标题所在页码
    ]

    你的任务不是分析完整层级结构，而是**仅判断每个标题是否为“一级标题”**。

    【输出规则】
    - 如果你认为该标题是一级标题，输出 1
    - 如果不是一级标题（属于二级及以下标题，或目录项、噪声项），输出 0
    - 所有输入元素都必须有输出，不允许遗漏
    - 输出字典的 key 必须与输入完全一致（包括 key 类型和数量）
    - 只允许输出 0 或 1，不要输出其他数字

    ━━━━━━━━━━━━━━━━━━
    【最高优先级否决规则（先判断）】
    ━━━━━━━━━━━━━━━━━━
    ⚠️ 只要满足以下任一条件，**必须直接输出 0**，不允许再按一级标题规则判断：

    1. 明显属于“目录 / 目次”中的条目，例如：
    - 标题中包含连续点线或符号用于连接页码（如“……”“......”“····”）
    - 标题末尾或中部包含页码标识，如 “(1)”“（3）”“57”
    - 标题结构类似 “编号 + 标题文本 + 页码”

    示例（全部判 0）：
    - 1 总 则 ……………………………… (1)
    - 2 术语、符号 ………………………… (3)
    - 5 民用建筑 (57)

    2. 标题文本明显是目录页中的子条目，而非正文标题
    3. 明显的 OCR 噪声或无语义文本（仅数字、空白等）

    ━━━━━━━━━━━━━━━━━━
    【一级标题判定规则（仅在未触发否决规则时适用）】
    ━━━━━━━━━━━━━━━━━━

    满足以下任一条件即可判为 1：

    1. 文档结构性标题：
    - 文档名称（如“某某技术规程 / 某某规范 / Technical code of practice …”）
    - 目次（仅“目次”二字本身）
    - 前言（不包含章节编号、页码）
    - 参考文献

    2. 规范正文的一级章节标题：
    - 范围
    - 规范性引用文件 / 引用文件
    - 术语 / 术语和定义
    - 基本规定 / 一般规定 / ××规定 / 总则
    - 不依附于其他章节的独立管理类标题（如：人员管理、设备管理、安全管理等）

    3. 形式特征辅助判断：
    - 行高明显大于正文或下级标题
    - 标题编号为单一整数（如“1 范围”“4 基本规定”）
    - 不包含小数编号（如 4.1、5.2、7.3 等）

    ━━━━━━━━━━━━━━━━━━
    【必须判为 0 的情况（补充）】
    ━━━━━━━━━━━━━━━━━━
    - 含有小数编号的标题（如 4.1、5.2、9.3 等）
    - 明显隶属于某个章节的子标题
    - 仅描述功能要求、控制项、具体设备、具体子系统的标题

    ━━━━━━━━━━━━━━━━━━
    【输出格式】
    ━━━━━━━━━━━━━━━━━━
    仅返回如下格式的字典，不要附加任何解释或说明：

    {
    {
    "0": 1,
    "1": 1,
    "2": 0,
    "3": 0
    }
    }

    Input title list:
    {title_dict}
    Corrected title list:
    """
    retry_count = 0
    max_retries = 3
    dict_completion = None

    # Build API call parameters
    api_params = {
        "model": title_aided_config["model"],
        "messages": [{'role': 'user', 'content': title_optimize_prompt}],
        "temperature": 0.7,
        "stream": True,
    }

    # Only add extra_body when explicitly specified in config
    if "enable_thinking" in title_aided_config:
        api_params["extra_body"] = {"enable_thinking": title_aided_config["enable_thinking"]}

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(**api_params)
            content_pieces = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_pieces.append(chunk.choices[0].delta.content)
            content = "".join(content_pieces).strip()
            # logger.info(f"Title completion: {content}")
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()
            dict_completion = json_repair.loads(content)
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}

            # logger.info(f"len(dict_completion): {len(dict_completion)}, len(title_dict): {len(title_dict)}")
            if len(dict_completion) == len(title_dict):
                for i, origin_title_block in enumerate(origin_title_list):
                    origin_title_block["level"] = int(dict_completion[i])
                break
            else:
                logger.warning(
                    "The number of titles in the optimized result is not equal to the number of titles in the input.")
                retry_count += 1
        except Exception as e:
            logger.exception(e)
            retry_count += 1

    if dict_completion is None:
        logger.error("Failed to decode dict after maximum retries.")
