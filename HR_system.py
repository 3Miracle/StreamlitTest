#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三峡物资招标公司干部画像管理系统
三峡物资招标公司数字化中心|智慧识才解决方案

核心功能：
1. 智能五维画像 - 基于NLP的个人材料分析
2. 江河胜任图模型 - 三峡物资招投标特色能力模型
3. 智能人岗匹配 - AI驱动的岗位适配分析
4. 数据可视化分析 - 多维度展示和分析
5. 人才发展建议 - 个性化发展路径规划

重构特点：
- 实现基于NLP的个人材料分析
- 优化江河胜任图模型实际应用
- 增强数据可视化和用户体验
- 提供实用的人才管理功能

作者: 三峡物资招标公司数字化中心
版本: 5.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
import logging
import math
import random
from itertools import combinations
import re
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import scipy.stats as stats
import jieba
import jieba.posseg as pseg
from textblob import TextBlob
# import spacy
from transformers import pipeline

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="三峡物资招标公司干部画像管理系统",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 优化后的全局样式
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== 核心数据模型 =====

class CadreProfile:
    """干部画像数据模型"""
    
    def __init__(self, name: str, age: int, gender: str, education: str, 
                 department: str, position: str, work_years: int, 
                 political_status: str, description: str = "", 
                 school: str = "", major: str = ""):
        self.name = name
        self.age = age
        self.gender = gender
        self.education = education
        self.school = school
        self.major = major
        self.department = department
        self.position = position
        self.work_years = work_years
        self.political_status = political_status
        self.description = description
        
        # 五维画像分数（将通过NLP分析获得）
        self.quality_foundation = 0
        self.competency = 0
        self.performance = 0
        self.self_portrait = 0
        self.reputation_score = 0
        
        # 江河胜任图能力分数
        self.river_competency = {}
        
        # 个人发展维度（岗位锻炼、党建工作、领导力、工作年限、职位职级）
        self.development_dimensions = {
            "岗位锻炼": 0,
            "党建工作": 0, 
            "领导力": 0,
            "工作年限": 0,
            "职位职级": 0
        }
        
        # 详细标签（从描述中提取）
        self.extracted_tags = {}
        
        # 人岗匹配分数
        self.position_match_scores = {}
        
        # 职业生涯成长轨迹
        self.career_milestones = []  # 职业里程碑
        self.awards_qualifications = {}  # 奖项、职称、资质等
        self.performance_history = []  # 绩效历史
        self.training_records = []  # 培训记录
        self.project_achievements = []  # 项目成就
        
        # 个人亲属关系
        self.family_relations = {
            "father": {"name": "", "work_unit": "", "job_position": ""},
            "mother": {"name": "", "work_unit": "", "job_position": ""},
            "spouse": {"name": "", "work_unit": "", "job_position": ""},
            "children": []  # [{"name": "", "work_unit": "", "job_position": ""}, ...]
        }

class PersonalMaterialAnalyzer:
    """个人材料分析器 - 基于NLP的五维画像提取"""
    
    def __init__(self):
        # 初始化中文分词
        # jieba.load_userdict("custom_dict.txt")  # 可加载自定义词典
        
        # 五维画像关键词词典
        self.dimension_keywords = {
            "素质基础": {
                "政治素养": ["政治立场", "政治觉悟", "理论学习", "党性修养", "政治敏锐性"],
                "道德品质": ["品德高尚", "诚实守信", "公正廉洁", "职业道德", "社会责任"],
                "文化素养": ["文化底蕴", "知识面广", "人文素养", "文化修养", "学识渊博"],
                "身心素质": ["身体健康", "心理健康", "精神饱满", "体魄强健", "心态良好"],
                "学习能力": ["学习主动", "接受新知", "自我提升", "持续学习", "知识更新"],
                "价值观念": ["价值导向", "人生观", "世界观", "价值观", "道德观念"]
            },
            "胜任能力": {
                "专业能力": ["专业知识", "业务精通", "技术能力", "专业水平", "业务素质"],
                "管理能力": ["组织管理", "团队建设", "统筹协调", "资源配置", "管理水平"],
                "创新能力": ["创新思维", "开拓创新", "改革创新", "创新实践", "创新成果"],
                "沟通能力": ["沟通协调", "表达能力", "人际交往", "协调能力", "沟通技巧"],
                "决策能力": ["决策果断", "分析判断", "科学决策", "决策水平", "判断力"]
            },
            "工作绩效": {
                "工作业绩": ["工作成绩", "业绩突出", "成果显著", "业绩优秀", "工作成效"],
                "工作效率": ["工作效率", "执行力强", "高效完成", "时间管理", "效率提升"],
                "工作质量": ["工作质量", "精益求精", "标准化", "质量控制", "工作精细"],
                "团队贡献": ["团队协作", "团队精神", "协作配合", "团队建设", "集体荣誉"],
                "创新成果": ["创新项目", "改进成果", "技术创新", "管理创新", "成果转化"]
            },
            "自画像": {
                "自我认知": ["自我认识", "自知之明", "优势认识", "不足反思", "自我评价"],
                "职业规划": ["职业目标", "发展规划", "职业发展", "目标明确", "规划合理"],
                "发展意愿": ["发展意愿", "上进心", "进取精神", "发展动力", "成长愿望"],
                "价值观": ["价值理念", "工作态度", "行为准则", "价值追求", "人生理念"],
                "工作态度": ["工作态度", "敬业精神", "责任心", "主动性", "积极性"]
            },
            "声誉得分": {
                "正面评价": ["群众认可", "领导肯定", "同事好评", "口碑良好", "声誉优秀"],
                "负面信息": ["违纪违法", "工作失误", "群众意见", "作风问题", "负面影响"],
                "廉洁表现": ["廉洁自律", "清正廉洁", "公私分明", "廉政建设", "反腐倡廉"],
                "群众评价": ["群众满意", "民主测评", "群众反映", "满意度", "群众基础"],
                "社会影响": ["社会影响", "社会评价", "公众形象", "社会责任", "影响力"]
            }
        }
        
        # 情感分析模型
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                              model="uer/roberta-base-finetuned-chinanews-chinese")
        except:
            self.sentiment_analyzer = None
            
    def extract_keywords_from_text(self, text: str) -> Dict[str, List[str]]:
        """从文本中提取关键词"""
        if not text:
            return {}
            
        # 分词和词性标注
        words = pseg.cut(text)
        
        # 提取有意义的词汇
        meaningful_words = []
        for word, flag in words:
            if len(word) > 1 and flag in ['n', 'v', 'a', 'nr', 'ns', 'nt']:
                meaningful_words.append(word)
        
        # 按维度分类关键词
        extracted = {}
        for dimension, categories in self.dimension_keywords.items():
            extracted[dimension] = {}
            for category, keywords in categories.items():
                matched_keywords = []
                for keyword in keywords:
                    if keyword in text or any(k in text for k in keyword.split()):
                        matched_keywords.append(keyword)
                extracted[dimension][category] = matched_keywords
                        
        return extracted
    
    def calculate_dimension_scores(self, text: str) -> Dict[str, float]:
        """计算五维画像分数"""
        if not text:
            return {
                "素质基础": 70,
                "胜任能力": 70,
                "工作绩效": 70,
                "自画像": 70,
                "声誉得分": 70
            }
        
        scores = {}
        
        # 文本长度影响基础分
        text_length_factor = min(len(text) / 500, 1.0)
        base_score = 60 + text_length_factor * 20
        
        for dimension, categories in self.dimension_keywords.items():
            dimension_score = base_score
            
            # 关键词匹配得分
            keyword_matches = 0
            total_keywords = 0
            
            for category, keywords in categories.items():
                total_keywords += len(keywords)
                for keyword in keywords:
                    if keyword in text:
                        keyword_matches += 1
                        
            # 关键词匹配率影响得分
            if total_keywords > 0:
                match_rate = keyword_matches / total_keywords
                dimension_score += match_rate * 25
            
            # 情感分析影响得分
            if self.sentiment_analyzer:
                try:
                    sentiment_result = self.sentiment_analyzer(text[:500])
                    if sentiment_result[0]['label'] == 'POSITIVE':
                        dimension_score += sentiment_result[0]['score'] * 10
                    else:
                        dimension_score -= sentiment_result[0]['score'] * 5
                except:
                    pass
            
            # 特殊处理声誉得分（负面信息降分）
            if dimension == "声誉得分":
                negative_keywords = ["违纪", "违法", "失误", "问题", "投诉", "批评"]
                for neg_keyword in negative_keywords:
                    if neg_keyword in text:
                        dimension_score -= 10
                        
            scores[dimension] = max(0, min(100, dimension_score))
        
        return scores
    
    def analyze_personal_material(self, cadre: CadreProfile) -> CadreProfile:
        """分析个人材料并更新画像"""
        # 提取关键词
        cadre.extracted_tags = self.extract_keywords_from_text(cadre.description)
        
        # 计算五维画像分数
        scores = self.calculate_dimension_scores(cadre.description)
        cadre.quality_foundation = scores["素质基础"]
        cadre.competency = scores["胜任能力"]
        cadre.performance = scores["工作绩效"]
        cadre.self_portrait = scores["自画像"]
        cadre.reputation_score = scores["声誉得分"]
        
        return cadre

class RiverCompetencyModel:
    """江河胜任图模型 - 三峡物资招投标特色能力模型"""
    
    def __init__(self):
        self.competency_dimensions = {
            "招投标管理能力": {
                "招标策划": {"权重": 0.25, "描述": "招标方案设计、需求分析、风险评估"},
                "投标管理": {"权重": 0.25, "描述": "投标文件审核、供应商管理、合规检查"},
                "评标专业": {"权重": 0.20, "描述": "评标标准制定、专家管理、评标流程"},
                "合同管理": {"权重": 0.15, "描述": "合同谈判、执行监督、变更管理"},
                "法规遵循": {"权重": 0.15, "描述": "法律法规掌握、合规审查、风险防控"}
            },
            "物资管理能力": {
                "采购策略": {"权重": 0.30, "描述": "采购计划、供应商选择、成本控制"},
                "库存管理": {"权重": 0.25, "描述": "库存优化、仓储管理、供应链协调"},
                "质量控制": {"权重": 0.20, "描述": "质量标准、检验检测、质量改进"},
                "物流协调": {"权重": 0.15, "描述": "运输配送、物流优化、配送管理"},
                "成本控制": {"权重": 0.10, "描述": "成本分析、预算控制、效益评估"}
            },
            "采购专业能力": {
                "采购计划": {"权重": 0.25, "描述": "采购需求分析、计划制定、预算管控"},
                "供应商管理": {"权重": 0.25, "描述": "供应商选择、评估、关系维护、绩效管理"},
                "合同谈判": {"权重": 0.20, "描述": "价格谈判、条款制定、风险控制"},
                "质量保证": {"权重": 0.15, "描述": "质量标准制定、验收管理、持续改进"},
                "成本优化": {"权重": 0.15, "描述": "成本分析、降本增效、价值工程"}
            },
            "风险管理能力": {
                "风险识别": {"权重": 0.25, "描述": "风险预警、隐患排查、风险评估"},
                "风险防控": {"权重": 0.25, "描述": "防控措施、应急预案、风险监控"},
                "合规管理": {"权重": 0.20, "描述": "制度建设、合规审查、监督检查"},
                "应急处理": {"权重": 0.15, "描述": "应急响应、危机处理、恢复管理"},
                "内控建设": {"权重": 0.15, "描述": "内控制度、流程优化、监督机制"}
            },
            "技术创新能力": {
                "技术研发": {"权重": 0.30, "描述": "技术研究、产品开发、创新设计"},
                "数字化应用": {"权重": 0.25, "描述": "信息化建设、数据分析、智能化应用"},
                "流程优化": {"权重": 0.20, "描述": "流程改进、效率提升、标准化建设"},
                "知识管理": {"权重": 0.15, "描述": "知识积累、经验总结、知识分享"},
                "持续改进": {"权重": 0.10, "描述": "持续改进、创新实践、成果转化"}
            },
            "大水电业务能力": {
                "水电工程": {"权重": 0.30, "描述": "水电站建设、设备选型、工程管理"},
                "技术标准": {"权重": 0.25, "描述": "技术规范、标准制定、质量控制"},
                "安全管理": {"权重": 0.20, "描述": "安全规程、风险防控、事故预防"},
                "环保要求": {"权重": 0.15, "描述": "环境保护、生态修复、绿色发展"},
                "运维管理": {"权重": 0.10, "描述": "设备维护、运行优化、寿命管理"}
            },
            "长江大保护能力": {
                "生态治理": {"权重": 0.30, "描述": "生态修复、环境治理、污染防控"},
                "项目管理": {"权重": 0.25, "描述": "项目策划、实施管理、绩效评估"},
                "政策理解": {"权重": 0.20, "描述": "政策解读、法规遵循、标准执行"},
                "技术应用": {"权重": 0.15, "描述": "环保技术、创新应用、效果评估"},
                "协调配合": {"权重": 0.10, "描述": "多方协调、资源配置、沟通合作"}
            },
            "抽水蓄能能力": {
                "蓄能技术": {"权重": 0.30, "描述": "抽水蓄能原理、技术特点、设备选型"},
                "电网调节": {"权重": 0.25, "描述": "电网调峰、频率调节、备用服务"},
                "运行管理": {"权重": 0.20, "描述": "运行策略、调度管理、效率优化"},
                "维护技术": {"权重": 0.15, "描述": "设备维护、技改升级、寿命延长"},
                "经济效益": {"权重": 0.10, "描述": "效益分析、成本控制、价值创造"}
            },
            "新能源业务能力": {
                "新能源技术": {"权重": 0.30, "描述": "风电、光伏、储能等新能源技术"},
                "项目开发": {"权重": 0.25, "描述": "项目选址、可研分析、投资决策"},
                "运营管理": {"权重": 0.20, "描述": "运营优化、发电管理、效率提升"},
                "并网技术": {"权重": 0.15, "描述": "并网标准、电能质量、系统稳定"},
                "政策应用": {"权重": 0.10, "描述": "政策解读、补贴申请、合规管理"}
            },
            "电子商城能力": {
                "平台运营": {"权重": 0.30, "描述": "平台管理、用户运营、商品管理"},
                "技术支持": {"权重": 0.25, "描述": "系统维护、技术优化、功能升级"},
                "供应链整合": {"权重": 0.20, "描述": "供应商整合、商品采购、库存管理"},
                "数据分析": {"权重": 0.15, "描述": "数据挖掘、用户行为分析、业务洞察"},
                "客户服务": {"权重": 0.10, "描述": "客户支持、售后服务、满意度提升"}
            }
        }
        
        # 能力等级标准
        self.competency_levels = {
            "初级": {"分数区间": (0, 60), "描述": "基础能力，需要指导"},
            "中级": {"分数区间": (60, 75), "描述": "独立工作，部分指导"},
            "高级": {"分数区间": (75, 90), "描述": "专业熟练，可指导他人"},
            "专家": {"分数区间": (90, 100), "描述": "专业权威，行业领先"}
        }
    
    def calculate_river_competency(self, cadre: CadreProfile) -> Dict[str, float]:
        """计算江河胜任图能力分数"""
        competency_scores = {}
        
        # 基于个人描述和五维画像计算能力分数
        for dimension, capabilities in self.competency_dimensions.items():
            dimension_score = 0
            
            for capability, config in capabilities.items():
                capability_score = self._calculate_capability_score(
                    cadre, capability, config["描述"]
                )
                dimension_score += capability_score * config["权重"]
            
            competency_scores[dimension] = dimension_score
        
        return competency_scores
    
    def calculate_development_dimensions(self, cadre: CadreProfile) -> Dict[str, float]:
        """计算个人发展维度分数"""
        dimensions = {}
        
        # 1. 岗位锻炼：基于工作经验、部门轮岗、职业发展轨迹
        position_training = 60  # 基础分
        position_training += min(cadre.work_years * 2, 30)  # 工作年限加分，最高30分
        
        # 基于职位级别调整
        position_levels = {
            "主管": 8, "经理": 12, "总监": 15, "部长": 18, "副总": 20,
            "专员": 3, "助理": 1, "员工": 0, "科员": 2
        }
        for level, bonus in position_levels.items():
            if level in cadre.position:
                position_training += bonus
                break
        
        dimensions["岗位锻炼"] = max(40, min(100, position_training))
        
        # 2. 党建工作：基于政治身份、理论学习、党务工作经验
        party_work = 50  # 基础分
        if "中共党员" in cadre.political_status:
            party_work += 25
        elif "预备党员" in cadre.political_status:
            party_work += 15
        elif "共青团员" in cadre.political_status:
            party_work += 10
        
        # 基于年龄和党龄推算
        if cadre.age > 35 and "党员" in cadre.political_status:
            party_work += 10  # 资深党员加分
        
        dimensions["党建工作"] = max(30, min(100, party_work))
        
        # 3. 领导力：基于管理经验、团队规模、领导职位
        leadership = 55  # 基础分
        
        # 基于职位级别判断领导力
        if any(title in cadre.position for title in ["主管", "经理", "总监", "部长", "副总", "主任"]):
            leadership += 20
        elif any(title in cadre.position for title in ["专员", "工程师", "分析师"]):
            leadership += 10
        
        # 基于工作年限增加管理经验
        if cadre.work_years >= 10:
            leadership += 15
        elif cadre.work_years >= 5:
            leadership += 8
        
        dimensions["领导力"] = max(35, min(100, leadership))
        
        # 4. 工作年限：直接基于实际工作年限计算
        work_years_score = min(cadre.work_years * 3.5, 100)  # 每年3.5分，最高100分
        if cadre.work_years >= 20:
            work_years_score = 95 + (cadre.work_years - 20) * 0.5  # 20年以上缓慢增长
        
        dimensions["工作年限"] = max(20, min(100, work_years_score))
        
        # 5. 职位职级：基于当前职位级别和发展潜力
        position_level = 50  # 基础分
        
        # 职位级别评分
        senior_positions = ["总监", "部长", "副总", "总经理"]
        middle_positions = ["主管", "经理", "主任", "科长"]
        junior_positions = ["专员", "工程师", "分析师", "助理"]
        
        if any(pos in cadre.position for pos in senior_positions):
            position_level += 35
        elif any(pos in cadre.position for pos in middle_positions):
            position_level += 25
        elif any(pos in cadre.position for pos in junior_positions):
            position_level += 15
        else:
            position_level += 5
        
        # 基于教育背景调整
        education_bonus = {"博士": 10, "硕士": 8, "本科": 5, "专科": 2}
        position_level += education_bonus.get(cadre.education, 0)
        
        dimensions["职位职级"] = max(30, min(100, position_level))
        
        return dimensions
    
    def _calculate_capability_score(self, cadre: CadreProfile, capability: str, description: str) -> float:
        """计算单项能力分数 - 个性化评分算法"""
        base_score = 60
        
        # 1. 基于工作经验调整
        experience_factor = min(cadre.work_years / 20, 1.0)
        base_score += experience_factor * 20
        
        # 2. 基于教育背景调整
        education_bonus = {
            "博士": 12,
            "硕士": 8,
            "本科": 5,
            "专科": 2
        }
        base_score += education_bonus.get(cadre.education, 0)
        
        # 3. 基于五维画像调整
        avg_score = (cadre.quality_foundation + cadre.competency + 
                    cadre.performance + cadre.self_portrait + 
                    cadre.reputation_score) / 5
        base_score += (avg_score - 70) * 0.4
        
        # 4. 基于岗位和个人特征的个性化调整
        personal_adjustment = self._get_personal_capability_adjustment(cadre, capability)
        base_score += personal_adjustment
        
        # 5. 基于部门专业匹配度调整
        department_bonus = self._get_department_capability_bonus(cadre, capability)
        base_score += department_bonus
        
        # 6. 基于职位级别调整
        position_bonus = self._get_position_level_bonus(cadre, capability)
        base_score += position_bonus
        
        # 7. 添加个体差异性 (基于姓名哈希确保一致性)
        import hashlib
        name_hash = int(hashlib.md5(f"{cadre.name}_{capability}".encode()).hexdigest()[:8], 16)
        individual_variance = (name_hash % 21 - 10)  # -10到+10的范围
        base_score += individual_variance
        
        return max(40, min(100, base_score))
    
    def _get_personal_capability_adjustment(self, cadre: CadreProfile, capability: str) -> float:
        """基于个人特征的能力调整"""
        adjustment = 0
        
        # 个人专业背景匹配
        specialty_keywords = {
            "招标策划": ["招标", "策划", "方案", "设计", "需求分析"],
            "投标管理": ["投标", "文件", "审核", "管理", "合规"],
            "评标专业": ["评标", "专家", "标准", "评审", "技术"],
            "合同管理": ["合同", "谈判", "执行", "法律", "条款"],
            "法规遵循": ["法规", "法律", "合规", "风险", "审查"],
            "采购策略": ["采购", "策略", "计划", "供应商", "成本"],
            "库存管理": ["库存", "仓储", "管理", "优化", "周转"],
            "质量控制": ["质量", "控制", "标准", "检验", "改进"],
            "物流协调": ["物流", "运输", "配送", "协调", "优化"],
            "成本控制": ["成本", "控制", "分析", "预算", "效益"],
            "风险识别": ["风险", "识别", "预警", "评估", "隐患"],
            "风险防控": ["防控", "措施", "预案", "监控", "应急"],
            "合规管理": ["合规", "制度", "审查", "监督", "检查"],
            "应急处理": ["应急", "响应", "危机", "处理", "恢复"],
            "内控建设": ["内控", "制度", "流程", "监督", "机制"],
            "技术研发": ["技术", "研发", "创新", "开发", "设计"],
            "数字化应用": ["数字化", "信息化", "智能化", "系统", "数据"],
            "流程优化": ["流程", "优化", "改进", "效率", "标准化"],
            "知识管理": ["知识", "管理", "积累", "分享", "经验"],
            "持续改进": ["改进", "创新", "实践", "转化", "优化"]
        }
        
        if capability in specialty_keywords:
            keywords = specialty_keywords[capability]
            match_count = sum(1 for keyword in keywords if keyword in cadre.description)
            adjustment += match_count * 3  # 每个匹配关键词+3分
        
        return adjustment
    
    def _get_department_capability_bonus(self, cadre: CadreProfile, capability: str) -> float:
        """基于部门专业性的能力加分"""
        department_capability_map = {
            "招投标管理部": ["招标策划", "投标管理", "评标专业", "合同管理", "法规遵循"],
            "物资管理部": ["采购策略", "库存管理", "质量控制", "物流协调", "成本控制"],
            "采购管理部": ["采购计划", "供应商管理", "合同谈判", "质量保证", "成本优化"],
            "风险管理部": ["风险识别", "风险防控", "合规管理", "应急处理", "内控建设"],
            "技术创新部": ["技术研发", "数字化应用", "流程优化", "知识管理", "持续改进"],
            "大水电事业部": ["水电工程", "技术标准", "安全管理", "环保要求", "运维管理"],
            "长江大保护部": ["生态治理", "项目管理", "政策理解", "技术应用", "协调配合"],
            "抽水蓄能部": ["蓄能技术", "电网调节", "运行管理", "维护技术", "经济效益"],
            "新能源事业部": ["新能源技术", "项目开发", "运营管理", "并网技术", "政策应用"],
            "电子商城部": ["平台运营", "技术支持", "供应链整合", "数据分析", "客户服务"],
            "供应链管理部": ["采购策略", "库存管理", "物流协调", "供应商管理", "成本控制"],
            "综合管理部": ["组织协调", "人员管理", "财务管理", "行政管理", "综合服务"]
        }
        
        if cadre.department in department_capability_map:
            if capability in department_capability_map[cadre.department]:
                return 8  # 专业匹配加8分
        
        return 0
    
    def _get_position_level_bonus(self, cadre: CadreProfile, capability: str) -> float:
        """基于职位级别的能力加分"""
        if "主管" in cadre.position or "经理" in cadre.position:
            return 5  # 管理岗位加5分
        elif "专员" in cadre.position or "工程师" in cadre.position:
            return 2  # 专业岗位加2分
        
        return 0
    
    def get_competency_level(self, score: float) -> str:
        """获取能力等级"""
        for level, config in self.competency_levels.items():
            if config["分数区间"][0] <= score < config["分数区间"][1]:
                return level
        return "专家"

class PositionMatchingEngine:
    """岗位匹配引擎"""
    
    def __init__(self):
        self.position_requirements = {
            "招投标专员": {
                "核心能力": ["招投标管理能力", "风险管理能力"],
                "最低要求": {"工作经验": 3, "教育": "本科", "五维最低": 70},
                "优先条件": {"专业背景": ["法律", "工程", "管理"], "证书": ["招投标师", "采购师"]}
            },
            "物资管理员": {
                "核心能力": ["物资管理能力", "风险管理能力"],
                "最低要求": {"工作经验": 2, "教育": "专科", "五维最低": 65},
                "优先条件": {"专业背景": ["物流", "管理", "工程"], "证书": ["物流师", "采购师"]}
            },
            "采购专员": {
                "核心能力": ["采购专业能力", "物资管理能力"],
                "最低要求": {"工作经验": 3, "教育": "本科", "五维最低": 70},
                "优先条件": {"专业背景": ["采购", "管理", "经济"], "证书": ["采购师", "供应链管理师"]}
            },
            "采购主管": {
                "核心能力": ["采购专业能力", "风险管理能力"],
                "最低要求": {"工作经验": 5, "教育": "本科", "五维最低": 75},
                "优先条件": {"专业背景": ["采购", "管理", "经济"], "证书": ["高级采购师", "供应链管理师"]}
            },
            "技术创新主管": {
                "核心能力": ["技术创新能力", "招投标管理能力"],
                "最低要求": {"工作经验": 5, "教育": "本科", "五维最低": 75},
                "优先条件": {"专业背景": ["计算机", "信息", "工程"], "证书": ["项目管理师"]}
            },
            "风险管理专员": {
                "核心能力": ["风险管理能力", "物资管理能力"],
                "最低要求": {"工作经验": 3, "教育": "本科", "五维最低": 72},
                "优先条件": {"专业背景": ["审计", "法律", "管理"], "证书": ["内审师", "风险管理师"]}
            },
            "大水电项目经理": {
                "核心能力": ["大水电业务能力", "采购专业能力"],
                "最低要求": {"工作经验": 8, "教育": "本科", "五维最低": 80},
                "优先条件": {"专业背景": ["水利", "电力", "工程"], "证书": ["一级建造师", "项目管理师"]}
            },
            "长江大保护专员": {
                "核心能力": ["长江大保护能力", "风险管理能力"],
                "最低要求": {"工作经验": 5, "教育": "本科", "五维最低": 75},
                "优先条件": {"专业背景": ["环境", "生态", "工程"], "证书": ["环评工程师", "环保工程师"]}
            },
            "抽水蓄能工程师": {
                "核心能力": ["抽水蓄能能力", "技术创新能力"],
                "最低要求": {"工作经验": 5, "教育": "本科", "五维最低": 75},
                "优先条件": {"专业背景": ["电力", "水利", "机械"], "证书": ["注册电气工程师", "项目管理师"]}
            },
            "新能源项目专员": {
                "核心能力": ["新能源业务能力", "采购专业能力"],
                "最低要求": {"工作经验": 4, "教育": "本科", "五维最低": 72},
                "优先条件": {"专业背景": ["新能源", "电力", "环境"], "证书": ["新能源工程师", "项目管理师"]}
            },
            "电子商城运营专员": {
                "核心能力": ["电子商城能力", "技术创新能力"],
                "最低要求": {"工作经验": 3, "教育": "本科", "五维最低": 70},
                "优先条件": {"专业背景": ["电子商务", "计算机", "管理"], "证书": ["电子商务师", "数据分析师"]}
            },
            "供应链管理主管": {
                "核心能力": ["物资管理能力", "采购专业能力"],
                "最低要求": {"工作经验": 6, "教育": "本科", "五维最低": 75},
                "优先条件": {"专业背景": ["供应链", "物流", "管理"], "证书": ["供应链管理师", "物流师"]}
            }
        }
    
    def calculate_position_match(self, cadre: CadreProfile, position: str) -> Dict[str, Any]:
        """计算岗位匹配度"""
        if position not in self.position_requirements:
            return {"匹配度": 0, "建议": "未找到该岗位要求"}
        
        requirements = self.position_requirements[position]
        match_score = 0
        match_details = {}
        recommendations = []
        
        # 1. 核心能力匹配 (40%)
        core_competency_score = 0
        for competency in requirements["核心能力"]:
            if competency in cadre.river_competency:
                core_competency_score += cadre.river_competency[competency]
        
        if requirements["核心能力"]:
            core_competency_score /= len(requirements["核心能力"])
        
        match_score += core_competency_score * 0.4
        match_details["核心能力匹配"] = core_competency_score
        
        # 2. 基础要求匹配 (30%)
        basic_score = 0
        
        # 工作经验
        if cadre.work_years >= requirements["最低要求"]["工作经验"]:
            basic_score += 25
        else:
            recommendations.append(f"需要{requirements['最低要求']['工作经验']}年以上工作经验")
        
        # 教育背景
        education_levels = {"专科": 1, "本科": 2, "硕士": 3, "博士": 4}
        if education_levels.get(cadre.education, 0) >= education_levels.get(requirements["最低要求"]["教育"], 0):
            basic_score += 25
        else:
            recommendations.append(f"需要{requirements['最低要求']['教育']}以上学历")
        
        # 五维画像
        avg_five_dimension = (cadre.quality_foundation + cadre.competency + 
                             cadre.performance + cadre.self_portrait + 
                             cadre.reputation_score) / 5
        if avg_five_dimension >= requirements["最低要求"]["五维最低"]:
            basic_score += 25
        else:
            recommendations.append(f"五维画像平均分需达到{requirements['最低要求']['五维最低']}分以上")
        
        match_score += basic_score * 0.3
        match_details["基础要求匹配"] = basic_score
        
        # 3. 五维画像匹配 (30%)
        five_dimension_score = avg_five_dimension
        match_score += five_dimension_score * 0.3
        match_details["五维画像匹配"] = five_dimension_score
        
        # 生成综合评价
        if match_score >= 85:
            overall_assessment = "高度匹配"
        elif match_score >= 70:
            overall_assessment = "基本匹配"
        elif match_score >= 60:
            overall_assessment = "需要提升"
        else:
            overall_assessment = "不匹配"
        
        return {
            "匹配度": match_score,
            "匹配详情": match_details,
            "综合评价": overall_assessment,
            "改进建议": recommendations
        }

class KnowledgeGraph:
    """知识图谱 - 干部关系网络分析与风险预防"""
    
    def __init__(self):
        # 干部-岗位关系（任职经历）
        self.cadre_position_history = defaultdict(list)
        
        # 干部-项目关系（参与情况）
        self.cadre_project_relations = defaultdict(list)
        
        # 项目-供应商关系（合作记录）
        self.project_supplier_relations = defaultdict(list)
        
        # 风险评估记录
        self.risk_assessments = {}
        
        # 初始化示例数据
        self.load_sample_relations()
    
    def load_sample_relations(self):
        """加载示例关系数据"""
        # 干部-岗位任职经历 (部分重点干部的详细经历)
        self.cadre_position_history["张明"] = [
            {"岗位": "招投标专员", "开始时间": "2012-07", "结束时间": "2015-06", "部门": "招投标管理部"},
            {"岗位": "招投标主管", "开始时间": "2015-07", "结束时间": "2018-12", "部门": "招投标管理部"},
            {"岗位": "招投标主管", "开始时间": "2019-01", "结束时间": "至今", "部门": "招投标管理部"}
        ]
        
        self.cadre_position_history["李华"] = [
            {"岗位": "物资管理员", "开始时间": "2008-07", "结束时间": "2012-06", "部门": "物资管理部"},
            {"岗位": "物资管理专员", "开始时间": "2012-07", "结束时间": "2016-12", "部门": "物资管理部"},
            {"岗位": "物资管理专员", "开始时间": "2017-01", "结束时间": "至今", "部门": "物资管理部"}
        ]
        
        self.cadre_position_history["王强"] = [
            {"岗位": "采购专员", "开始时间": "2013-07", "结束时间": "2016-06", "部门": "采购管理部"},
            {"岗位": "采购主管", "开始时间": "2016-07", "结束时间": "至今", "部门": "采购管理部"}
        ]
        
        self.cadre_position_history["陈科技"] = [
            {"岗位": "系统开发工程师", "开始时间": "2014-07", "结束时间": "2017-12", "部门": "技术创新部"},
            {"岗位": "技术创新主管", "开始时间": "2018-01", "结束时间": "至今", "部门": "技术创新部"}
        ]
        
        self.cadre_position_history["李水电"] = [
            {"岗位": "水电工程师", "开始时间": "2002-07", "结束时间": "2008-12", "部门": "大水电事业部"},
            {"岗位": "项目副经理", "开始时间": "2009-01", "结束时间": "2015-06", "部门": "大水电事业部"},
            {"岗位": "大水电项目经理", "开始时间": "2015-07", "结束时间": "至今", "部门": "大水电事业部"}
        ]
        
        self.cadre_position_history["链条长"] = [
            {"岗位": "物流专员", "开始时间": "2003-07", "结束时间": "2008-12", "部门": "物资管理部"},
            {"岗位": "供应链专员", "开始时间": "2009-01", "结束时间": "2015-06", "部门": "供应链管理部"},
            {"岗位": "供应链管理主管", "开始时间": "2015-07", "结束时间": "至今", "部门": "供应链管理部"}
        ]
        
        # 补充更多干部的任职经历
        self.cadre_position_history["陈雅静"] = [
            {"岗位": "法务助理", "开始时间": "2019-07", "结束时间": "2021-06", "部门": "招投标管理部"},
            {"岗位": "招投标专员", "开始时间": "2021-07", "结束时间": "至今", "部门": "招投标管理部"}
        ]
        
        self.cadre_position_history["刘建国"] = [
            {"岗位": "工程师", "开始时间": "2006-07", "结束时间": "2012-12", "部门": "技术中心"},
            {"岗位": "高级工程师", "开始时间": "2013-01", "结束时间": "2018-06", "部门": "招投标管理部"},
            {"岗位": "评标专家", "开始时间": "2018-07", "结束时间": "至今", "部门": "招投标管理部"}
        ]
        
        self.cadre_position_history["赵国强"] = [
            {"岗位": "仓库管理员", "开始时间": "2012-07", "结束时间": "2016-12", "部门": "物资管理部"},
            {"岗位": "仓储管理主管", "开始时间": "2017-01", "结束时间": "至今", "部门": "物资管理部"}
        ]
        
        self.cadre_position_history["李雪莲"] = [
            {"岗位": "国际贸易专员", "开始时间": "2016-07", "结束时间": "2019-12", "部门": "采购管理部"},
            {"岗位": "采购专员", "开始时间": "2020-01", "结束时间": "至今", "部门": "采购管理部"}
        ]
        
        self.cadre_position_history["周智慧"] = [
            {"岗位": "软件开发工程师", "开始时间": "2019-07", "结束时间": "2022-06", "部门": "技术创新部"},
            {"岗位": "系统开发工程师", "开始时间": "2022-07", "结束时间": "至今", "部门": "技术创新部"}
        ]
        
        self.cadre_position_history["刘风控"] = [
            {"岗位": "审计专员", "开始时间": "2006-07", "结束时间": "2012-12", "部门": "审计部"},
            {"岗位": "风险管理专员", "开始时间": "2013-01", "结束时间": "至今", "部门": "风险管理部"}
        ]
        
        self.cadre_position_history["王大坝"] = [
            {"岗位": "水利工程师", "开始时间": "2007-07", "结束时间": "2015-12", "部门": "工程部"},
            {"岗位": "水电工程师", "开始时间": "2016-01", "结束时间": "至今", "部门": "大水电事业部"}
        ]
        
        self.cadre_position_history["周生态"] = [
            {"岗位": "环保工程师", "开始时间": "2012-07", "结束时间": "2017-12", "部门": "环保部"},
            {"岗位": "长江大保护专员", "开始时间": "2018-01", "结束时间": "至今", "部门": "长江大保护部"}
        ]
        
        self.cadre_position_history["杨调峰"] = [
            {"岗位": "电气工程师", "开始时间": "2014-07", "结束时间": "2019-06", "部门": "电力部"},
            {"岗位": "电网工程师", "开始时间": "2019-07", "结束时间": "至今", "部门": "抽水蓄能部"}
        ]
        
        self.cadre_position_history["风光明"] = [
            {"岗位": "新能源工程师", "开始时间": "2013-07", "结束时间": "2018-12", "部门": "新能源部"},
            {"岗位": "新能源项目专员", "开始时间": "2019-01", "结束时间": "至今", "部门": "新能源事业部"}
        ]
        
        self.cadre_position_history["商小城"] = [
            {"岗位": "电商运营专员", "开始时间": "2018-07", "结束时间": "2021-12", "部门": "信息中心"},
            {"岗位": "电子商城运营专员", "开始时间": "2022-01", "结束时间": "至今", "部门": "电子商城部"}
        ]
        
        # 干部-项目参与情况 (覆盖各业务板块)
        self.cadre_project_relations["张明"] = [
            {"项目": "三峡左岸电站设备采购", "角色": "招标负责人", "参与时间": "2015-03至2015-12", "项目金额": "15亿元"},
            {"项目": "溪洛渡水电站物资采购", "角色": "招标主管", "参与时间": "2017-06至2018-05", "项目金额": "8.5亿元"},
            {"项目": "乌东德水电站建设物资", "角色": "项目负责人", "参与时间": "2019-01至2021-12", "项目金额": "12亿元"},
            {"项目": "白鹤滩水电站设备招标", "角色": "总协调人", "参与时间": "2020-06至2022-08", "项目金额": "20亿元"}
        ]
        
        self.cadre_project_relations["李华"] = [
            {"项目": "三峡左岸电站设备采购", "角色": "物资协调员", "参与时间": "2015-03至2015-12", "项目金额": "15亿元"},
            {"项目": "长江大保护物资供应", "角色": "物资管理负责人", "参与时间": "2018-01至2020-12", "项目金额": "6亿元"},
            {"项目": "抽水蓄能电站物资", "角色": "供应链管理", "参与时间": "2021-01至2023-06", "项目金额": "9亿元"}
        ]
        
        self.cadre_project_relations["王强"] = [
            {"项目": "大型设备集中采购", "角色": "采购主管", "参与时间": "2016-01至2018-12", "项目金额": "25亿元"},
            {"项目": "国际设备进口采购", "角色": "项目经理", "参与时间": "2019-01至2021-06", "项目金额": "18亿元"},
            {"项目": "新能源设备采购", "角色": "采购负责人", "参与时间": "2021-07至2023-12", "项目金额": "30亿元"}
        ]
        
        self.cadre_project_relations["陈科技"] = [
            {"项目": "智能招投标系统开发", "角色": "技术负责人", "参与时间": "2018-01至2020-12", "项目金额": "1.2亿元"},
            {"项目": "数字化物资管理平台", "角色": "项目经理", "参与时间": "2021-01至2022-06", "项目金额": "0.8亿元"},
            {"项目": "AI智能采购系统", "角色": "总设计师", "参与时间": "2022-07至2024-06", "项目金额": "1.5亿元"}
        ]
        
        self.cadre_project_relations["李水电"] = [
            {"项目": "三峡水利枢纽工程", "角色": "物资采购总监", "参与时间": "2005-01至2012-12", "项目金额": "180亿元"},
            {"项目": "溪洛渡水电站建设", "角色": "项目经理", "参与时间": "2013-01至2018-12", "项目金额": "120亿元"},
            {"项目": "白鹤滩水电站建设", "角色": "大水电项目经理", "参与时间": "2019-01至2022-12", "项目金额": "200亿元"}
        ]
        
        self.cadre_project_relations["周生态"] = [
            {"项目": "长江大保护生态修复", "角色": "生态专家", "参与时间": "2018-01至2021-12", "项目金额": "45亿元"},
            {"项目": "三峡库区生态治理", "角色": "项目负责人", "参与时间": "2020-01至2023-06", "项目金额": "28亿元"},
            {"项目": "长江流域污染防控", "角色": "技术总监", "参与时间": "2021-07至2024-12", "项目金额": "35亿元"}
        ]
        
        self.cadre_project_relations["马蓄能"] = [
            {"项目": "丰宁抽水蓄能电站", "角色": "技术负责人", "参与时间": "2015-01至2019-12", "项目金额": "192亿元"},
            {"项目": "浙江长龙山抽蓄电站", "角色": "工程师", "参与时间": "2018-01至2021-06", "项目金额": "73亿元"},
            {"项目": "河北易县抽蓄电站", "角色": "项目经理", "参与时间": "2021-07至2024-12", "项目金额": "87亿元"}
        ]
        
        self.cadre_project_relations["风光明"] = [
            {"项目": "青海海西新能源基地", "角色": "项目经理", "参与时间": "2019-01至2022-12", "项目金额": "150亿元"},
            {"项目": "内蒙古乌兰察布风电", "角色": "技术负责人", "参与时间": "2020-06至2023-06", "项目金额": "95亿元"},
            {"项目": "甘肃酒泉光伏发电", "角色": "工程总监", "参与时间": "2022-01至2024-12", "项目金额": "120亿元"}
        ]
        
        self.cadre_project_relations["商小城"] = [
            {"项目": "三峡电子商城平台", "角色": "运营经理", "参与时间": "2020-01至2023-12", "项目金额": "3.5亿元"},
            {"项目": "B2B物资交易平台", "角色": "产品经理", "参与时间": "2021-06至2024-06", "项目金额": "2.8亿元"},
            {"项目": "智慧供应链平台", "角色": "运营负责人", "参与时间": "2022-01至2024-12", "项目金额": "4.2亿元"}
        ]
        
        self.cadre_project_relations["链条长"] = [
            {"项目": "全国供应链网络优化", "角色": "供应链总监", "参与时间": "2018-01至2021-12", "项目金额": "8.5亿元"},
            {"项目": "智慧物流体系建设", "角色": "项目经理", "参与时间": "2020-01至2023-06", "项目金额": "6.2亿元"},
            {"项目": "绿色供应链示范工程", "角色": "技术负责人", "参与时间": "2022-01至2024-12", "项目金额": "5.8亿元"}
        ]
        
        # 补充更多干部的项目参与
        self.cadre_project_relations["陈雅静"] = [
            {"项目": "招投标法律合规审查", "角色": "法务专员", "参与时间": "2021-01至2023-12", "项目金额": "0.5亿元"},
            {"项目": "供应商资质审核系统", "角色": "合规审查员", "参与时间": "2022-06至2024-06", "项目金额": "0.3亿元"}
        ]
        
        self.cadre_project_relations["刘建国"] = [
            {"项目": "三峡左岸电站设备采购", "角色": "技术评标专家", "参与时间": "2015-03至2015-12", "项目金额": "15亿元"},
            {"项目": "白鹤滩水电站设备招标", "角色": "首席评标专家", "参与时间": "2020-06至2022-08", "项目金额": "20亿元"},
            {"项目": "抽水蓄能设备技术评审", "角色": "技术顾问", "参与时间": "2021-01至2023-12", "项目金额": "12亿元"}
        ]
        
        self.cadre_project_relations["赵国强"] = [
            {"项目": "智能仓储系统建设", "角色": "仓储负责人", "参与时间": "2018-01至2020-12", "项目金额": "1.5亿元"},
            {"项目": "物资管理标准化项目", "角色": "实施经理", "参与时间": "2020-01至2022-06", "项目金额": "0.8亿元"}
        ]
        
        self.cadre_project_relations["李雪莲"] = [
            {"项目": "国际设备进口采购", "角色": "国际采购专员", "参与时间": "2017-01至2019-12", "项目金额": "18亿元"},
            {"项目": "欧洲风电设备采购", "角色": "采购项目经理", "参与时间": "2021-01至2023-06", "项目金额": "8.5亿元"}
        ]
        
        self.cadre_project_relations["周智慧"] = [
            {"项目": "数字化物资管理平台", "角色": "前端开发工程师", "参与时间": "2021-01至2022-06", "项目金额": "0.8亿元"},
            {"项目": "AI智能采购系统", "角色": "系统架构师", "参与时间": "2022-07至2024-06", "项目金额": "1.5亿元"}
        ]
        
        self.cadre_project_relations["刘风控"] = [
            {"项目": "内控体系建设项目", "角色": "风控负责人", "参与时间": "2015-01至2018-12", "项目金额": "0.6亿元"},
            {"项目": "供应商风险评估系统", "角色": "风险管理专家", "参与时间": "2019-01至2021-12", "项目金额": "0.4亿元"},
            {"项目": "廉洁风险预警平台", "角色": "项目经理", "参与时间": "2022-01至2024-12", "项目金额": "0.7亿元"}
        ]
        
        self.cadre_project_relations["王大坝"] = [
            {"项目": "溪洛渡水电站建设", "角色": "水电工程师", "参与时间": "2013-01至2018-12", "项目金额": "120亿元"},
            {"项目": "乌东德水电站工程", "角色": "技术专家", "参与时间": "2016-01至2020-12", "项目金额": "180亿元"},
            {"项目": "白鹤滩水电站建设", "角色": "工程技术顾问", "参与时间": "2019-01至2022-12", "项目金额": "200亿元"}
        ]
        
        self.cadre_project_relations["赵风电"] = [
            {"项目": "内蒙古乌兰察布风电", "角色": "风电工程师", "参与时间": "2020-06至2023-06", "项目金额": "95亿元"},
            {"项目": "江苏海上风电项目", "角色": "海上风电专家", "参与时间": "2021-01至2024-06", "项目金额": "68亿元"}
        ]
        
        self.cadre_project_relations["钱光伏"] = [
            {"项目": "甘肃酒泉光伏发电", "角色": "光伏工程师", "参与时间": "2022-01至2024-12", "项目金额": "120亿元"},
            {"项目": "分布式光伏示范项目", "角色": "技术负责人", "参与时间": "2021-06至2023-12", "项目金额": "15亿元"}
        ]
        
        self.cadre_project_relations["李平台"] = [
            {"项目": "三峡电子商城平台", "角色": "技术开发工程师", "参与时间": "2020-01至2023-12", "项目金额": "3.5亿元"},
            {"项目": "移动端商城应用", "角色": "移动开发负责人", "参与时间": "2022-01至2024-06", "项目金额": "1.2亿元"}
        ]
        
        self.cadre_project_relations["流程优"] = [
            {"项目": "供应链流程再造", "角色": "流程优化专家", "参与时间": "2019-01至2021-12", "项目金额": "2.5亿元"},
            {"项目": "采购流程标准化", "角色": "项目经理", "参与时间": "2021-01至2023-06", "项目金额": "1.8亿元"}
        ]
        
        # 项目-供应商合作记录 (覆盖各类型项目)
        self.project_supplier_relations["三峡左岸电站设备采购"] = [
            {"供应商": "东方电气集团", "合作内容": "发电机组", "合同金额": "8亿元", "评价": "优秀"},
            {"供应商": "哈尔滨电气集团", "合作内容": "水轮机", "合同金额": "4亿元", "评价": "良好"},
            {"供应商": "上海电气集团", "合作内容": "电力设备", "合同金额": "3亿元", "评价": "良好"}
        ]
        
        self.project_supplier_relations["溪洛渡水电站物资采购"] = [
            {"供应商": "中国一重", "合作内容": "重型设备", "合同金额": "3.5亿元", "评价": "优秀"},
            {"供应商": "宝钢集团", "合作内容": "钢材供应", "合同金额": "2亿元", "评价": "良好"},
            {"供应商": "三一重工", "合作内容": "工程机械", "合同金额": "3亿元", "评价": "优秀"}
        ]
        
        self.project_supplier_relations["白鹤滩水电站建设"] = [
            {"供应商": "东方电气集团", "合作内容": "发电机组", "合同金额": "85亿元", "评价": "优秀"},
            {"供应商": "哈尔滨电气集团", "合作内容": "水轮发电机", "合同金额": "70亿元", "评价": "优秀"},
            {"供应商": "中国安能", "合作内容": "土建工程", "合同金额": "45亿元", "评价": "良好"}
        ]
        
        self.project_supplier_relations["大型设备集中采购"] = [
            {"供应商": "国电龙源", "合作内容": "风电设备", "合同金额": "12亿元", "评价": "优秀"},
            {"供应商": "金风科技", "合作内容": "风力发电机", "合同金额": "8亿元", "评价": "优秀"},
            {"供应商": "明阳智能", "合作内容": "海上风电", "合同金额": "5亿元", "评价": "良好"}
        ]
        
        self.project_supplier_relations["长江大保护生态修复"] = [
            {"供应商": "中国节能", "合作内容": "环保设备", "合同金额": "15亿元", "评价": "优秀"},
            {"供应商": "光大环境", "合作内容": "污水处理", "合同金额": "12亿元", "评价": "良好"},
            {"供应商": "启迪环境", "合作内容": "固废处理", "合同金额": "8亿元", "评价": "良好"},
            {"供应商": "碧水源", "合作内容": "水处理技术", "合同金额": "10亿元", "评价": "优秀"}
        ]
        
        self.project_supplier_relations["丰宁抽水蓄能电站"] = [
            {"供应商": "东方电气集团", "合作内容": "抽蓄机组", "合同金额": "60亿元", "评价": "优秀"},
            {"供应商": "哈尔滨电气集团", "合作内容": "发电电机", "合同金额": "45亿元", "评价": "优秀"},
            {"供应商": "沃尔沃建筑设备", "合作内容": "施工机械", "合同金额": "8亿元", "评价": "良好"},
            {"供应商": "中国建筑", "合作内容": "土建施工", "合同金额": "79亿元", "评价": "良好"}
        ]
        
        self.project_supplier_relations["青海海西新能源基地"] = [
            {"供应商": "隆基绿能", "合作内容": "光伏组件", "合同金额": "45亿元", "评价": "优秀"},
            {"供应商": "晶科能源", "合作内容": "太阳能电池", "合同金额": "35亿元", "评价": "优秀"},
            {"供应商": "阳光电源", "合作内容": "逆变器设备", "合同金额": "28亿元", "评价": "良好"},
            {"供应商": "特变电工", "合作内容": "变压器", "合同金额": "22亿元", "评价": "良好"},
            {"供应商": "上海电气", "合作内容": "风电机组", "合同金额": "20亿元", "评价": "优秀"}
        ]
        
        self.project_supplier_relations["三峡电子商城平台"] = [
            {"供应商": "阿里云", "合作内容": "云计算服务", "合同金额": "1.2亿元", "评价": "优秀"},
            {"供应商": "腾讯云", "合作内容": "云存储服务", "合同金额": "0.8亿元", "评价": "良好"},
            {"供应商": "华为", "合作内容": "IT基础设施", "合同金额": "1.5亿元", "评价": "优秀"}
        ]
        
        self.project_supplier_relations["智能招投标系统开发"] = [
            {"供应商": "用友网络", "合作内容": "ERP系统", "合同金额": "0.4亿元", "评价": "良好"},
            {"供应商": "金蝶软件", "合作内容": "财务系统", "合同金额": "0.3亿元", "评价": "良好"},
            {"供应商": "浪潮集团", "合作内容": "服务器设备", "合同金额": "0.5亿元", "评价": "优秀"}
        ]
        
        self.project_supplier_relations["全国供应链网络优化"] = [
            {"供应商": "中远海运", "合作内容": "物流运输", "合同金额": "3.2亿元", "评价": "优秀"},
            {"供应商": "顺丰控股", "合作内容": "快递物流", "合同金额": "1.8亿元", "评价": "优秀"},
            {"供应商": "京东物流", "合作内容": "仓储配送", "合同金额": "2.5亿元", "评价": "良好"},
            {"供应商": "菜鸟网络", "合作内容": "智慧物流", "合同金额": "1亿元", "评价": "良好"}
        ]
    
    def analyze_integrity_risks(self, cadre_name: str) -> Dict[str, Any]:
        """分析干部廉洁合规风险"""
        risks = {
            "风险等级": "低风险",
            "风险因素": [],
            "预警信息": [],
            "建议措施": []
        }
        
        # 分析任职经历风险
        position_history = self.cadre_position_history.get(cadre_name, [])
        project_relations = self.cadre_project_relations.get(cadre_name, [])
        
        # 1. 频繁岗位变动风险
        if len(position_history) > 4:
            risks["风险因素"].append("任职经历变动频繁")
            risks["预警信息"].append("建议关注岗位适应性")
        
        # 2. 长期同一岗位风险
        current_position_years = 0
        if position_history:
            latest_position = position_history[-1]
            if latest_position["结束时间"] == "至今":
                start_year = int(latest_position["开始时间"][:4])
                current_position_years = datetime.now().year - start_year
                if current_position_years > 8:
                    risks["风险因素"].append("同一岗位任职时间过长")
                    risks["预警信息"].append("建议适时轮岗交流")
        
        # 3. 项目参与度风险分析
        if project_relations:
            total_amount = 0
            high_value_projects = 0
            
            for project in project_relations:
                amount_str = project["项目金额"].replace("亿元", "")
                amount = float(amount_str)
                total_amount += amount
                
                if amount >= 10:  # 10亿以上为高价值项目
                    high_value_projects += 1
            
            if high_value_projects >= 2:
                risks["风险因素"].append("多个高价值项目参与")
                risks["预警信息"].append("建议加强重大项目监督")
            
            if total_amount >= 30:  # 总项目金额30亿以上
                risks["风险因素"].append("累计项目金额巨大")
                risks["预警信息"].append("建议定期审计检查")
        
        # 4. 供应商关系风险
        supplier_risk_score = self._analyze_supplier_relationships(cadre_name)
        if supplier_risk_score > 0.6:
            risks["风险因素"].append("供应商关系密切度较高")
            risks["预警信息"].append("建议加强供应商关系管理")
        
        # 综合风险等级评估
        risk_factor_count = len(risks["风险因素"])
        if risk_factor_count >= 3:
            risks["风险等级"] = "高风险"
            risks["建议措施"].extend([
                "立即启动专项审计",
                "加强日常监督检查",
                "建立重点关注机制"
            ])
        elif risk_factor_count >= 2:
            risks["风险等级"] = "中风险"
            risks["建议措施"].extend([
                "定期开展廉政谈话",
                "加强制度学习培训",
                "适当调整工作安排"
            ])
        else:
            risks["建议措施"].extend([
                "保持现有监督机制",
                "继续廉政教育",
                "定期风险评估"
            ])
        
        return risks
    
    def _analyze_supplier_relationships(self, cadre_name: str) -> float:
        """分析与供应商关系密切度"""
        project_relations = self.cadre_project_relations.get(cadre_name, [])
        
        if not project_relations:
            return 0.0
        
        # 统计供应商合作次数
        supplier_interactions = defaultdict(int)
        
        for project in project_relations:
            project_name = project["项目"]
            suppliers = self.project_supplier_relations.get(project_name, [])
            
            for supplier in suppliers:
                supplier_name = supplier["供应商"]
                supplier_interactions[supplier_name] += 1
        
        # 计算风险评分
        total_interactions = sum(supplier_interactions.values())
        if total_interactions == 0:
            return 0.0
        
        # 如果与某个供应商合作次数过多，风险增加
        max_interactions = max(supplier_interactions.values()) if supplier_interactions else 0
        
        risk_score = min(max_interactions / len(project_relations), 1.0)
        
        return risk_score
    
    def get_relationship_network(self, cadre_name: str) -> Dict[str, Any]:
        """获取干部关系网络图数据"""
        network_data = {
            "nodes": [],
            "edges": [],
            "center_node": cadre_name
        }
        
        # 获取干部详细信息
        target_cadre = None
        for cadre in self.cadres if hasattr(self, 'cadres') else []:
            if cadre.name == cadre_name:
                target_cadre = cadre
                break
        
        # 添加中心节点（干部）
        cadre_label = f"{cadre_name}"
        if target_cadre:
            # 根据职位确定职级
            if any(pos in target_cadre.position for pos in ["总监", "部长", "副总"]):
                job_level = "高级"
            elif any(pos in target_cadre.position for pos in ["主管", "经理", "主任"]):
                job_level = "中级" 
            else:
                job_level = "初级"
            cadre_label = f"{cadre_name}\\n({job_level})"
        
        network_data["nodes"].append({
            "id": cadre_name,
            "label": cadre_label,
            "type": "cadre",
            "size": 35,
            "color": "#FF6B6B",
            "details": {
                "name": cadre_name,
                "job_level": job_level if target_cadre else "未知",
                "department": target_cadre.department if target_cadre else "未知",
                "position": target_cadre.position if target_cadre else "未知"
            }
        })
        
        # 个人工作关系网络仅包含项目和供应商关联，不显示岗位信息
        # 添加项目节点和关系
        project_relations = self.cadre_project_relations.get(cadre_name, [])
        for i, project in enumerate(project_relations):
            project_id = f"proj_{i}_{project['项目']}"
            # 生成招标编号和预算
            tender_number = f"ZB{2023}{(i+1):03d}"
            budget = project.get('项目金额', '未知')
            
            network_data["nodes"].append({
                "id": project_id,
                "label": f"项目{i+1}",  # 简化显示
                "type": "project",
                "size": 25,
                "color": "#45B7D1",
                "hover_info": f"项目：{project['项目'][:15]}...<br>预算：{budget}<br>角色：{project['角色']}",
                "details": {
                    "project_name": project['项目'],
                    "tender_number": tender_number,
                    "budget": budget,
                    "participation_time": project['参与时间'],
                    "role": project['角色'],
                    "description": f"招标编号：{tender_number}<br>项目预算：{budget}<br>参与时间：{project['参与时间']}<br>担任角色：{project['角色']}"
                }
            })
            
            network_data["edges"].append({
                "source": cadre_name,
                "target": project_id,
                "label": "参与情况",
                "color": "#45B7D1",
                "details": {
                    "relation_type": "参与情况", 
                    "role": project['角色'],
                    "time": project['参与时间']
                }
            })
            
            # 添加供应商节点和关系
            suppliers = self.project_supplier_relations.get(project['项目'], [])
            for j, supplier in enumerate(suppliers):
                supplier_id = f"supp_{i}_{j}_{supplier['供应商']}"
                # 生成供应商评级
                import random
                ratings = ["AAA", "AA", "A", "BBB", "BB"]
                rating = random.choice(ratings)
                
                network_data["nodes"].append({
                    "id": supplier_id,
                    "label": f"供应商{j+1}",  # 简化显示
                    "type": "supplier",
                    "size": 20,
                    "color": "#96CEB4",
                    "hover_info": f"供应商：{supplier['供应商'][:12]}...<br>评级：{rating}<br>合作：{supplier['合作内容'][:10]}...",
                    "details": {
                        "supplier_name": supplier['供应商'],
                        "rating": rating,
                        "cooperation_content": supplier['合作内容'],
                        "project_relation": project['项目'],
                        "description": f"供应商：{supplier['供应商']}<br>信用评级：{rating}<br>合作内容：{supplier['合作内容']}<br>关联项目：{project['项目']}"
                    }
                })
                
                network_data["edges"].append({
                    "source": project_id,
                    "target": supplier_id,
                    "label": "合作记录",
                    "color": "#96CEB4",
                    "details": {
                        "relation_type": "合作记录",
                        "content": supplier['合作内容'],
                        "supplier_rating": rating
                    }
                })
        
        # 添加个人亲属关系节点和关系
        # 这里需要获取干部信息来添加亲属关系
        # 简化处理：通过干部姓名查找对应的干部对象
        target_cadre = None
        for cadre in self.cadres if hasattr(self, 'cadres') else []:
            if cadre.name == cadre_name:
                target_cadre = cadre
                break
        
        if target_cadre and hasattr(target_cadre, 'family_relations'):
            # 添加父亲
            if target_cadre.family_relations["father"]["name"]:
                father_id = f"father_{cadre_name}"
                network_data["nodes"].append({
                    "id": father_id,
                    "label": f"父亲：{target_cadre.family_relations['father']['name']}\n{target_cadre.family_relations['father']['work_unit']}",
                    "type": "family",
                    "size": 18,
                    "color": "#8E44AD"
                })
                network_data["edges"].append({
                    "source": cadre_name,
                    "target": father_id,
                    "label": "父子关系",
                    "color": "#8E44AD"
                })
            
            # 添加母亲
            if target_cadre.family_relations["mother"]["name"]:
                mother_id = f"mother_{cadre_name}"
                network_data["nodes"].append({
                    "id": mother_id,
                    "label": f"母亲：{target_cadre.family_relations['mother']['name']}\n{target_cadre.family_relations['mother']['work_unit']}",
                    "type": "family",
                    "size": 18,
                    "color": "#E91E63"
                })
                network_data["edges"].append({
                    "source": cadre_name,
                    "target": mother_id,
                    "label": "母子关系",
                    "color": "#E91E63"
                })
            
            # 添加配偶
            if target_cadre.family_relations["spouse"]["name"]:
                spouse_id = f"spouse_{cadre_name}"
                network_data["nodes"].append({
                    "id": spouse_id,
                    "label": f"配偶：{target_cadre.family_relations['spouse']['name']}\n{target_cadre.family_relations['spouse']['work_unit']}",
                    "type": "family",
                    "size": 22,
                    "color": "#FF9800"
                })
                network_data["edges"].append({
                    "source": cadre_name,
                    "target": spouse_id,
                    "label": "夫妻关系",
                    "color": "#FF9800"
                })
            
            # 添加子女
            for i, child in enumerate(target_cadre.family_relations["children"]):
                if child["name"]:
                    child_id = f"child_{i}_{cadre_name}"
                    network_data["nodes"].append({
                        "id": child_id,
                        "label": f"子女：{child['name']}\n{child['work_unit']}",
                        "type": "family",
                        "size": 16,
                        "color": "#00BCD4"
                    })
                    network_data["edges"].append({
                        "source": cadre_name,
                        "target": child_id,
                        "label": "父子关系",
                        "color": "#00BCD4"
                    })
        
        return network_data

class DataManager:
    """数据管理器"""
    
    def __init__(self):
        self.cadres = []
        self.knowledge_graph = KnowledgeGraph()
        self.load_sample_data()
        # 为知识图谱设置干部数据引用
        self.knowledge_graph.cadres = self.cadres
    
    def load_sample_data(self):
        """加载50个多样化的示例数据"""
        sample_cadres = [
            # 招投标管理部 (8人)
            {
                "name": "张明",
                "age": 35,
                "gender": "男",
                "education": "本科",
                "school": "华中科技大学",
                "major": "工程管理",
                "department": "招投标管理部",
                "position": "招投标主管",
                "work_years": 8,
                "political_status": "中共党员",
                "description": "张明，男，汉族，1988年3月出生，湖北宜昌人，本科学历，工程管理学士学位，中级经济师，注册采购师。现任三峡物资招标公司招投标管理部主管。一、政治素质表现：政治立场坚定，理论学习扎实，党性修养较强，能够自觉用习近平新时代中国特色社会主义思想指导工作实践。积极参与党组织生活，主动承担党建工作任务，政治敏锐性和政治鉴别力较强。二、道德品行表现：品德高尚，诚实守信，公正廉洁，职业道德良好。严格遵守廉洁自律各项规定，公私分明，从未发生违法违纪行为。待人真诚，乐于助人，具有良好的人际关系和社会声誉。三、专业能力表现：专业知识扎实，招投标业务精通，熟练掌握《招标投标法》等相关法律法规。组织管理能力强，创新思维活跃，具有较强的战略思维和前瞻意识。主持完成大型招标项目80余项，涉及金额超过15亿元，项目成功率100%。建立供应商评估体系和风险预警机制，提高采购效率35%，降低采购成本8%。四、工作绩效表现：工作业绩突出，连续五年考核优秀，获得公司先进个人、优秀党员等荣誉称号。团队协作能力强，积极培养新人，所在团队连续三年获得先进集体。创新工作方法，推进招投标流程标准化，获得管理创新奖。五、个人作风表现：工作态度认真，责任心强，主动性高，敢于担当。注重学习提升，持续更新知识结构，积极参加各类专业培训。执行力强，能够高质量完成上级交办的各项任务。六、主要不足：需要加强跨部门协调能力，提升国际招投标经验，在复杂项目的统筹协调方面有待进一步提升。"
            },
            {
                "name": "陈雅静",
                "age": 29,
                "gender": "女", 
                "education": "硕士",
                "school": "中国政法大学",
                "major": "法学",
                "department": "招投标管理部",
                "position": "招投标专员",
                "work_years": 5,
                "political_status": "中共党员",
                "description": "陈雅静，女，汉族，1994年8月出生，江苏南京人，硕士研究生学历，法学硕士学位，中级经济师。政治觉悟高，理论基础扎实。专业法律知识扎实，招投标法规掌握全面，合规审查能力强。参与重大招标项目法律审核30余项，确保项目合规性。具有较强的文字表达和沟通协调能力。工作认真负责，注重细节，执行力强。需要加强项目管理经验。"
            },
            {
                "name": "刘建国",
                "age": 44,
                "gender": "男",
                "education": "本科", 
                "school": "山东大学",
                "major": "土木工程",
                "department": "招投标管理部",
                "position": "评标专家",
                "work_years": 18,
                "political_status": "中共党员",
                "description": "刘建国，男，汉族，1979年5月出生，山东济南人，本科学历，工程学士学位，高级工程师。政治立场坚定，党性修养深厚。专业技术能力突出，评标经验丰富，担任评标专家多年。参与各类评标工作200余次，评标标准制定科学合理。具有丰富的工程项目经验，技术判断准确。工作严谨细致，公正廉洁，业界声誉良好。需要更新数字化评标技能。"
            },
            {
                "name": "周晓敏",
                "age": 31,
                "gender": "女",
                "education": "硕士",
                "school": "中南大学",
                "major": "工商管理",
                "department": "招投标管理部", 
                "position": "合同管理员",
                "work_years": 7,
                "political_status": "共青团员",
                "description": "周晓敏，女，汉族，1992年11月出生，湖南株洲人，硕士研究生学历，工商管理硕士学位，中级经济师。政治素质良好，思想品德端正。合同管理专业能力强，法律条款把握准确，风险识别能力突出。负责重大合同谈判和管理工作，合同执行率达98%。具有较强的谈判技巧和沟通能力。工作主动积极，学习能力强。需要提升国际合同管理经验。"
            },
            {
                "name": "马志远",
                "age": 27,
                "gender": "男",
                "education": "本科",
                "school": "河北经贸大学",
                "major": "经济学",
                "department": "招投标管理部",
                "position": "招标专员",
                "work_years": 3,
                "political_status": "中共预备党员",
                "description": "马志远，男，汉族，1996年4月出生，河北石家庄人，本科学历，经济学学士学位，助理经济师。政治思想积极，入党动机端正。专业基础扎实，学习能力强，适应性好。参与招标项目策划和实施20余项，工作完成质量高。具有创新意识，积极运用新技术优化工作流程。工作态度认真，团队协作意识强。需要积累更多大型项目经验。"
            },
            {
                "name": "孙丽娟",
                "age": 39,
                "gender": "女",
                "education": "本科",
                "school": "大连理工大学",
                "major": "管理学",
                "department": "招投标管理部",
                "position": "供应商管理专员",
                "work_years": 14,
                "political_status": "中共党员",
                "description": "孙丽娟，女，汉族，1984年7月出生，辽宁大连人，本科学历，管理学学士学位，中级经济师。政治立场坚定，理论学习认真。供应商管理经验丰富，建立了完善的供应商评估体系。负责供应商准入、评价和退出管理，供应商满意度持续提升。具有较强的数据分析和风险识别能力。工作细致负责，执行力强。需要加强国际供应商管理经验。"
            },
            {
                "name": "徐海波",
                "age": 33,
                "gender": "男",
                "education": "硕士",
                "school": "浙江大学",
                "major": "工程管理",
                "department": "招投标管理部",
                "position": "招投标工程师",
                "work_years": 9,
                "political_status": "中共党员",
                "description": "徐海波，男，汉族，1990年9月出生，浙江杭州人，硕士研究生学历，工程管理硕士学位，中级工程师。政治觉悟高，理论联系实际能力强。工程技术背景深厚，招投标专业知识扎实。主持技术标评审工作，技术方案评价准确。具有较强的工程项目管理和协调能力。工作认真负责，创新意识强。需要加强跨专业技术领域知识。"
            },
            {
                "name": "韩雪梅",
                "age": 26,
                "gender": "女",
                "education": "本科",
                "school": "山西大学",
                "major": "法学",
                "department": "招投标管理部",
                "position": "投标文件审核员",
                "work_years": 2,
                "political_status": "共青团员",
                "description": "韩雪梅，女，汉族，1997年12月出生，山西太原人，本科学历，法学学士学位，助理经济师。政治思想进步，品德优良。法律专业基础扎实，文档审核能力强，细致认真。负责投标文件合规性审查，发现问题及时准确。具有较强的责任心和学习能力。工作态度端正，服务意识强。需要加强实际项目经验积累。"
            },

            # 物资管理部 (7人)
            {
                "name": "李华",
                "age": 42,
                "gender": "女",
                "education": "硕士",
                "school": "湖南大学",
                "major": "物流管理",
                "department": "物资管理部",
                "position": "物资管理专员",
                "work_years": 15,
                "political_status": "中共党员",
                "description": "李华，女，汉族，1981年6月出生，湖南长沙人，硕士研究生学历，物流管理硕士学位，高级经济师，注册物流师，供应链管理师。现任三峡物资招标公司物资管理部专员。一、政治素质表现：政治觉悟高，理论基础扎实，能够正确处理政治与业务的关系，自觉在思想上政治上行动上同党中央保持高度一致。积极参加理论学习，认真学习党的二十大精神，具有较强的政治责任感和使命感。二、道德品行表现：品德良好，待人诚恳，工作公正，廉洁自律。严格遵守各项纪律规定，清正廉洁，从不利用职务便利谋取私利。具有强烈的事业心和责任感，深受同事和领导信任。三、专业能力表现：物资管理专业能力强，熟悉供应链管理理论与实践，库存控制经验丰富，精通ERP系统操作。建立完善的物资管理制度体系，制定物资采购、储存、配送标准化流程，降低库存成本25%，提高库存周转率40%。具有较强的数据分析能力和流程优化能力，善于运用信息化手段提升管理效率。四、工作绩效表现：工作成效显著，连续四年考核优秀，获得部门先进工作者、公司优秀员工等荣誉。注重团队建设，培养了多名业务骨干，所管理的库存准确率达99.8%。主持实施物资管理信息化项目，获得公司科技进步三等奖。五、个人作风表现：工作细致认真，执行力强，能够承担重要任务。学习能力强，积极参加各类培训，不断更新专业知识。善于沟通协调，具有良好的团队合作精神。六、主要不足：需要提升技术创新能力，加强新技术应用；跨部门沟通协调有待加强，国际物流管理经验相对欠缺。"
            },
            {
                "name": "赵国强",
                "age": 36,
                "gender": "男",
                "education": "本科",
                "school": "郑州大学",
                "major": "物流管理",
                "department": "物资管理部",
                "position": "仓储管理主管",
                "work_years": 12,
                "political_status": "中共党员",
                "description": "赵国强，男，汉族，1987年3月出生，河南郑州人，本科学历，物流管理学士学位，中级经济师。政治立场坚定，工作作风扎实。仓储管理经验丰富，仓库运营效率高，库存准确率达99.5%。推进仓储信息化建设，提升管理水平。具有较强的团队管理和协调能力。工作认真负责，执行力强。需要加强智能仓储技术应用。"
            },
            {
                "name": "钱小芳",
                "age": 29,
                "gender": "女",
                "education": "本科",
                "school": "南昌大学",
                "major": "质量管理工程",
                "department": "物资管理部",
                "position": "质量控制专员",
                "work_years": 6,
                "political_status": "中共党员",
                "description": "钱小芳，女，汉族，1994年6月出生，江西南昌人，本科学历，质量管理学士学位，中级工程师。政治思想端正，职业道德良好。质量管理专业知识扎实，检验检测技能熟练。建立完善的质量控制体系，产品合格率持续提升。具有较强的问题分析和解决能力。工作严谨细致，责任心强。需要加强新技术检测方法学习。"
            },
            {
                "name": "吴建华",
                "age": 48,
                "gender": "男",
                "education": "专科",
                "school": "安徽职业技术学院",
                "major": "物流管理",
                "department": "物资管理部",
                "position": "物流协调员",
                "work_years": 24,
                "political_status": "中共党员",
                "description": "吴建华，男，汉族，1975年1月出生，安徽合肥人，专科学历，物流管理专业，中级经济师。政治觉悟高，党性修养深厚。物流运输经验丰富，熟悉各种运输方式和路线。协调物流配送工作，运输及时率达95%以上。具有丰富的实践经验和良好的协调能力。工作勤恳踏实，任劳任怨。需要学习现代物流信息技术。"
            },
            {
                "name": "林雅琪",
                "age": 30,
                "gender": "女",
                "education": "硕士",
                "school": "厦门大学",
                "major": "会计学",
                "department": "物资管理部",
                "position": "成本控制分析师",
                "work_years": 6,
                "political_status": "共青团员",
                "description": "林雅琪，女，汉族，1993年10月出生，福建厦门人，硕士研究生学历，会计学硕士学位，中级会计师。政治素质良好，品行端正。成本分析专业能力强，财务数据分析准确。建立成本控制模型，为决策提供数据支撑。具有较强的逻辑思维和数据分析能力。工作认真细致，学习能力强。需要加强业务成本深度理解。"
            },
            {
                "name": "黄志明",
                "age": 41,
                "gender": "男", 
                "education": "本科",
                "school": "华南理工大学",
                "major": "机械工程",
                "department": "物资管理部",
                "position": "设备管理员",
                "work_years": 17,
                "political_status": "中共党员",
                "description": "黄志明，男，汉族，1982年8月出生，广东广州人，本科学历，机械工程学士学位，高级工程师。政治立场坚定，思想觉悟高。设备管理专业技能扎实，维护保养经验丰富。负责重要设备的管理维护，设备完好率达98%。具有较强的技术分析和故障诊断能力。工作责任心强，技术水平高。需要学习智能设备管理技术。"
            },
            {
                "name": "陈美玲",
                "age": 34,
                "gender": "女",
                "education": "本科",
                "school": "西南交通大学",
                "major": "管理学",
                "department": "物资管理部",
                "position": "库存管理专员", 
                "work_years": 10,
                "political_status": "中共党员",
                "description": "陈美玲，女，汉族，1989年4月出生，四川成都人，本科学历，管理学学士学位，中级经济师。政治思想端正，工作作风优良。库存管理经验丰富，数据统计分析能力强。优化库存结构，提高周转率15%。具有较强的计划协调和执行能力。工作主动认真，团队合作意识强。需要加强智能库存管理系统应用。"
            },

            # 采购管理部 (6人)
            {
                "name": "王强",
                "age": 38,
                "gender": "男",
                "education": "博士",
                "school": "西南财经大学",
                "major": "管理学",
                "department": "采购管理部",
                "position": "采购主管",
                "work_years": 12,
                "political_status": "中共党员",
                "description": "王强，男，汉族，1985年9月出生，四川成都人，博士研究生学历，管理学博士学位，高级经济师，国际注册采购经理，项目管理专业人士(PMP)。现任三峡物资招标公司采购管理部主管。一、政治素质表现：政治立场坚定，理论水平高，能够将党的理论与采购管理实践相结合。积极参与党组织活动，主动学习贯彻习近平总书记关于国有企业改革发展的重要论述，政治敏锐性强。二、道德品行表现：品德高尚，诚实守信，严格遵守采购廉洁纪律，建立了完善的个人廉洁档案。坚持公开透明的采购原则，从未发生任何违纪违法行为，在业界享有良好声誉。三、专业能力表现：采购管理专业能力突出，战略采购经验丰富，精通国内外采购法规和惯例。创新采购模式，建立集中采购平台，实现采购成本节约和效率提升。主导完成重大采购项目200余项，累计采购金额超过100亿元，节约成本超过3.5亿元。具有较强的谈判技巧和供应商管理能力，建立了覆盖全球的优质供应商网络。四、工作绩效表现：工作成效显著，连续六年考核优秀，获得公司科技进步一等奖、管理创新奖等荣誉。主持制定采购管理制度15项，培训采购人员500余人次。所负责的采购项目质量合格率100%，供应商满意度达95%以上。五、个人作风表现：工作严谨务实，具有强烈的创新意识和进取精神。注重学习提升，每年参加国内外专业培训，发表管理类论文8篇。团队协作能力强，善于整合资源，推动跨部门协同。六、主要不足：需要加强国际采购经验，特别是'一带一路'沿线国家采购实践；在新兴技术采购领域的专业知识有待加强。"
            },
            {
                "name": "李雪莲",
                "age": 32,
                "gender": "女",
                "education": "硕士",
                "school": "对外经济贸易大学",
                "major": "国际贸易",
                "department": "采购管理部",
                "position": "采购专员",
                "work_years": 8,
                "political_status": "中共党员",
                "description": "李雪莲，女，汉族，1991年7月出生，北京市人，硕士研究生学历，国际贸易硕士学位，中级经济师。政治觉悟高，理论联系实际。国际采购经验丰富，外语能力强，熟悉国际贸易规则。负责进口设备采购，质量和进度控制良好。具有较强的跨文化沟通和协调能力。工作严谨负责，适应能力强。需要加强技术设备专业知识。"
            },
            {
                "name": "张伟民",
                "age": 45,
                "gender": "男",
                "education": "本科",
                "school": "天津大学",
                "major": "工商管理",
                "department": "采购管理部",
                "position": "供应商管理主管",
                "work_years": 20,
                "political_status": "中共党员",
                "description": "张伟民，男，汉族，1978年2月出生，天津市人，本科学历，工商管理学士学位，高级经济师。政治立场坚定，党性修养深厚。供应商管理经验丰富，关系维护能力强。建立完善的供应商评价体系，优化供应商结构。具有较强的商务谈判和风险控制能力。工作稳重可靠，业务能力突出。需要加强数字化供应商管理。"
            },
            {
                "name": "刘敏慧",
                "age": 28,
                "gender": "女",
                "education": "硕士",
                "school": "苏州大学",
                "major": "工程管理",
                "department": "采购管理部",
                "position": "采购工程师",
                "work_years": 4,
                "political_status": "共青团员",
                "description": "刘敏慧，女，汉族，1995年5月出生，江苏苏州人，硕士研究生学历，工程管理硕士学位，助理工程师。政治思想进步，品德优良。工程技术基础扎实，采购专业知识全面。参与技术设备采购项目，技术规格把握准确。具有较强的技术分析和方案评估能力。工作认真主动，学习能力强。需要加强实际工程经验。"
            },
            {
                "name": "孙志华",
                "age": 37,
                "gender": "男",
                "education": "本科",
                "school": "中国海洋大学",
                "major": "法学",
                "department": "采购管理部",
                "position": "合同谈判专员",
                "work_years": 13,
                "political_status": "中共党员",
                "description": "孙志华，男，汉族，1986年11月出生，山东青岛人，本科学历，法学学士学位，中级经济师。政治素质良好，法律意识强。合同谈判经验丰富，法律条款把握准确。主导重要合同谈判，为公司争取有利条件。具有较强的语言表达和逻辑思维能力。工作原则性强，谈判技巧娴熟。需要加强国际合同法律知识。"
            },
            {
                "name": "何晓燕",
                "age": 31,
                "gender": "女",
                "education": "本科",
                "school": "西南大学",
                "major": "经济学",
                "department": "采购管理部",
                "position": "价格分析师",
                "work_years": 7,
                "political_status": "中共党员",
                "description": "何晓燕，女，汉族，1992年9月出生，重庆市人，本科学历，经济学学士学位，中级经济师。政治思想端正，工作态度认真。价格分析专业能力强，市场信息收集全面。建立价格监测体系，为采购决策提供数据支撑。具有较强的数据分析和市场研判能力。工作细致严谨，责任心强。需要加强大数据分析技术应用。"
            },

            # 技术创新部 (5人)
            {
                "name": "陈科技",
                "age": 35,
                "gender": "男",
                "education": "博士",
                "school": "上海交通大学",
                "major": "计算机科学与技术",
                "department": "技术创新部",
                "position": "技术创新主管",
                "work_years": 10,
                "political_status": "中共党员",
                "description": "陈科技，男，汉族，1988年6月出生，上海市人，博士研究生学历，计算机科学与技术博士学位，正高级工程师。政治立场坚定，理论水平高。技术创新能力突出，主持开发智能招投标系统，获得软件著作权8项。推动公司数字化转型，提升管理效率。具有较强的项目管理和团队领导能力。工作严谨创新，成果转化能力强。需要加强产业化推广经验。"
            },
            {
                "name": "周智慧",
                "age": 29,
                "gender": "女",
                "education": "硕士",
                "school": "华中科技大学",
                "major": "软件工程",
                "department": "技术创新部",
                "position": "系统开发工程师",
                "work_years": 5,
                "political_status": "中共党员",
                "description": "周智慧，女，汉族，1994年3月出生，湖北武汉人，硕士研究生学历，软件工程硕士学位，中级工程师。政治觉悟高，思想品德端正。软件开发技术扎实，编程能力强。参与多个信息系统开发项目，代码质量高。具有较强的系统分析和设计能力。工作认真负责，技术钻研精神强。需要加强业务领域深度理解。"
            },
            {
                "name": "李数据",
                "age": 33,
                "gender": "男",
                "education": "硕士",
                "school": "中山大学",
                "major": "统计学",
                "department": "技术创新部",
                "position": "数据分析师",
                "work_years": 9,
                "political_status": "中共党员",
                "description": "李数据，男，汉族，1990年12月出生，广东深圳人，硕士研究生学历，统计学硕士学位，中级工程师。政治思想端正，工作作风务实。数据分析专业能力强，熟练掌握各种分析工具。建立业务数据模型，为决策提供科学依据。具有较强的逻辑思维和数据洞察能力。工作严谨细致，学习能力强。需要加强人工智能技术应用。"
            },
            {
                "name": "王创新",
                "age": 27,
                "gender": "男",
                "education": "本科",
                "school": "浙江大学",
                "major": "信息管理与信息系统",
                "department": "技术创新部",
                "position": "产品经理",
                "work_years": 3,
                "political_status": "共青团员",
                "description": "王创新，男，汉族，1996年8月出生，浙江杭州人，本科学历，信息管理学士学位，助理工程师。政治思想积极，创新意识强。产品设计思维活跃，用户体验关注度高。参与多个产品规划和设计项目，用户满意度良好。具有较强的沟通协调和项目推进能力。工作主动积极，适应能力强。需要积累更多行业经验。"
            },
            {
                "name": "张智能",
                "age": 40,
                "gender": "男",
                "education": "硕士",
                "school": "西安电子科技大学",
                "major": "人工智能",
                "department": "技术创新部",
                "position": "AI工程师",
                "work_years": 15,
                "political_status": "中共党员",
                "description": "张智能，男，汉族，1983年1月出生，陕西西安人，硕士研究生学历，人工智能硕士学位，高级工程师。政治立场坚定，技术视野开阔。人工智能技术专业能力强，机器学习算法精通。主导AI技术在招投标领域的应用，提升智能化水平。具有较强的技术创新和算法优化能力。工作专业精深，前瞻性强。需要加强团队管理能力。"
            },

            # 风险管理部 (4人)
            {
                "name": "刘风控",
                "age": 43,
                "gender": "男",
                "education": "硕士",
                "school": "南京大学",
                "major": "风险管理",
                "department": "风险管理部",
                "position": "风险管理专员",
                "work_years": 18,
                "political_status": "中共党员",
                "description": "刘风控，男，汉族，1980年4月出生，江苏南京人，硕士研究生学历，风险管理硕士学位，高级经济师。政治觉悟高，原则性强。风险管理专业能力突出，风险识别和评估经验丰富。建立完善的风险管控体系，有效防范各类风险。具有较强的分析判断和预警能力。工作严谨负责，执行力强。需要加强新兴风险领域研究。"
            },
            {
                "name": "赵合规",
                "age": 36,
                "gender": "女",
                "education": "硕士",
                "school": "中国政法大学",
                "major": "法学",
                "department": "风险管理部",
                "position": "合规审查员",
                "work_years": 12,
                "political_status": "中共党员",
                "description": "赵合规，女，汉族，1987年7月出生，河北石家庄人，硕士研究生学历，法学硕士学位，中级经济师。政治立场坚定，法律意识强。合规管理专业知识扎实，法规理解准确。负责各类业务合规审查，确保依法合规经营。具有较强的法律分析和风险识别能力。工作认真细致，责任心强。需要加强国际法律法规学习。"
            },
            {
                "name": "钱内控",
                "age": 39,
                "gender": "男",
                "education": "本科",
                "school": "山西财经大学",
                "major": "会计学",
                "department": "风险管理部",
                "position": "内控专员",
                "work_years": 15,
                "political_status": "中共党员",
                "description": "钱内控，男，汉族，1984年10月出生，山西太原人，本科学历，会计学学士学位，高级会计师。政治素质良好，职业操守严格。内控制度建设经验丰富，流程梳理能力强。完善内控管理体系，提升管理规范性。具有较强的制度设计和执行监督能力。工作原则性强，执行严格。需要加强数字化内控技术应用。"
            },
            {
                "name": "孙监督",
                "age": 32,
                "gender": "女",
                "education": "本科",
                "school": "东北财经大学",
                "major": "审计学",
                "department": "风险管理部",
                "position": "监督检查员",
                "work_years": 8,
                "political_status": "中共党员",
                "description": "孙监督，女，汉族，1991年2月出生，辽宁沈阳人，本科学历，审计学学士学位，中级审计师。政治思想端正，工作作风严谨。监督检查专业能力强，问题发现敏锐。开展各类专项检查，及时发现和纠正问题。具有较强的调查分析和报告撰写能力。工作认真负责，执行力强。需要加强大数据审计技术应用。"
            },

            # 大水电事业部 (5人)
            {
                "name": "李水电",
                "age": 46,
                "gender": "男",
                "education": "硕士",
                "school": "清华大学",
                "major": "水利工程",
                "department": "大水电事业部",
                "position": "大水电项目经理",
                "work_years": 22,
                "political_status": "中共党员",
                "description": "李水电，男，汉族，1977年5月出生，湖北宜昌人，硕士研究生学历，水利工程硕士学位，教授级高级工程师。政治立场坚定，党性修养深厚。水电工程专业技术精深，项目管理经验丰富。主持三峡、溪洛渡等重大水电项目物资采购，确保工程建设顺利进行。具有较强的工程技术和组织协调能力。工作责任心强，业绩突出。需要加强新能源技术融合应用。"
            },
            {
                "name": "王大坝",
                "age": 41,
                "gender": "男",
                "education": "本科",
                "school": "四川大学",
                "major": "水利水电工程",
                "department": "大水电事业部",
                "position": "水电工程师",
                "work_years": 17,
                "political_status": "中共党员",
                "description": "王大坝，男，汉族，1982年3月出生，四川乐山人，本科学历，水利水电工程学士学位，高级工程师。政治觉悟高，技术功底扎实。水电工程技术专业能力强，现场经验丰富。参与多个大型水电站建设项目，技术方案合理可行。具有较强的工程分析和技术创新能力。工作严谨认真，技术水平高。需要加强智能化技术应用。"
            },
            {
                "name": "张发电",
                "age": 35,
                "gender": "男",
                "education": "硕士",
                "school": "云南大学",
                "major": "电气工程",
                "department": "大水电事业部",
                "position": "设备工程师",
                "work_years": 11,
                "political_status": "中共党员",
                "description": "张发电，男，汉族，1988年8月出生，云南昆明人，硕士研究生学历，电气工程硕士学位，中级工程师。政治思想端正，专业技术过硬。电气设备专业知识扎实，设备选型经验丰富。负责水电站主要设备技术规格制定，设备性能优良。具有较强的技术分析和设备管理能力。工作认真负责，学习能力强。需要加强国际先进技术了解。"
            },
            {
                "name": "陈环保",
                "age": 38,
                "gender": "女",
                "education": "硕士",
                "school": "贵州大学",
                "major": "环境工程",
                "department": "大水电事业部",
                "position": "环保工程师",
                "work_years": 14,
                "political_status": "中共党员",
                "description": "陈环保，女，汉族，1985年6月出生，贵州贵阳人，硕士研究生学历，环境工程硕士学位，中级工程师。政治立场坚定，环保意识强。环境保护专业技术扎实，环评工作经验丰富。确保水电项目环保合规，生态保护措施有效。具有较强的环境分析和治理方案设计能力。工作严谨负责，社会责任感强。需要加强生态修复新技术学习。"
            },
            {
                "name": "刘安全",
                "age": 44,
                "gender": "男",
                "education": "本科",
                "school": "中南大学",
                "major": "安全工程",
                "department": "大水电事业部",
                "position": "安全工程师",
                "work_years": 20,
                "political_status": "中共党员",
                "description": "刘安全，男，汉族，1979年11月出生，湖南长沙人，本科学历，安全工程学士学位，高级工程师。政治觉悟高，安全责任意识强。安全管理专业能力突出，风险防控经验丰富。建立完善的安全管理体系，确保项目安全生产。具有较强的安全分析和应急处置能力。工作责任心强，执行严格。需要加强智能安全监控技术应用。"
            },

            # 长江大保护部 (4人)
            {
                "name": "周生态",
                "age": 37,
                "gender": "男",
                "education": "博士",
                "school": "南京大学",
                "major": "生态学",
                "department": "长江大保护部",
                "position": "长江大保护专员",
                "work_years": 12,
                "political_status": "中共党员",
                "description": "周生态，男，汉族，1986年4月出生，江苏镇江人，博士研究生学历，生态学博士学位，高级工程师。政治立场坚定，生态文明理念强。生态环境保护专业技术精深，长江生态修复经验丰富。主持长江大保护相关项目，生态效果显著。具有较强的生态分析和保护方案设计能力。工作严谨负责，社会担当强。需要加强政策法规深度理解。"
            },
            {
                "name": "吴治理",
                "age": 33,
                "gender": "女",
                "education": "硕士",
                "school": "安徽大学",
                "major": "环境科学",
                "department": "长江大保护部",
                "position": "环境治理工程师",
                "work_years": 9,
                "political_status": "中共党员",
                "description": "吴治理，女，汉族，1990年7月出生，安徽芜湖人，硕士研究生学历，环境科学硕士学位，中级工程师。政治觉悟高，环保责任感强。环境治理专业技术扎实，污染防控经验丰富。参与多个环境治理项目，治理效果良好。具有较强的环境监测和治理技术能力。工作认真负责，执行力强。需要加强新兴污染物治理技术学习。"
            },
            {
                "name": "胡修复",
                "age": 29,
                "gender": "男",
                "education": "硕士",
                "school": "中国地质大学",
                "major": "生态修复",
                "department": "长江大保护部",
                "position": "生态修复工程师",
                "work_years": 5,
                "political_status": "共青团员",
                "description": "胡修复，男，汉族，1994年9月出生，湖北荆州人，硕士研究生学历，生态修复硕士学位，助理工程师。政治思想进步，环保意识强。生态修复专业知识扎实，修复技术掌握全面。参与长江流域生态修复项目，修复效果明显。具有较强的生态分析和修复方案实施能力。工作主动积极，学习能力强。需要积累更多大型项目经验。"
            },
            {
                "name": "许协调",
                "age": 42,
                "gender": "女",
                "education": "本科",
                "school": "南昌大学",
                "major": "公共管理",
                "department": "长江大保护部",
                "position": "项目协调员",
                "work_years": 18,
                "political_status": "中共党员",
                "description": "许协调，女，汉族，1981年12月出生，江西九江人，本科学历，公共管理学士学位，高级经济师。政治立场坚定，大局意识强。项目协调管理经验丰富，沟通协调能力强。统筹各方资源，确保项目顺利实施。具有较强的组织协调和关系维护能力。工作稳重可靠，执行力强。需要加强技术业务知识学习。"
            },

            # 抽水蓄能部 (3人)
            {
                "name": "马蓄能",
                "age": 40,
                "gender": "男",
                "education": "硕士",
                "school": "华北电力大学",
                "major": "电力系统及其自动化",
                "department": "抽水蓄能部",
                "position": "抽水蓄能工程师",
                "work_years": 16,
                "political_status": "中共党员",
                "description": "马蓄能，男，汉族，1983年3月出生，内蒙古呼和浩特人，硕士研究生学历，电力系统硕士学位，高级工程师。政治觉悟高，技术水平精深。抽水蓄能技术专业能力强，电网调节经验丰富。参与多个抽水蓄能电站建设项目，技术方案先进可行。具有较强的电力系统分析和优化能力。工作严谨认真，技术创新意识强。需要加强新型储能技术融合应用。"
            },
            {
                "name": "杨调峰",
                "age": 34,
                "gender": "男",
                "education": "本科",
                "school": "兰州理工大学",
                "major": "电气工程及其自动化",
                "department": "抽水蓄能部",
                "position": "电网工程师",
                "work_years": 10,
                "political_status": "中共党员",
                "description": "杨调峰，男，汉族，1989年5月出生，甘肃兰州人，本科学历，电气工程学士学位，中级工程师。政治思想端正，专业基础扎实。电网运行专业知识全面，调峰调频技术熟练。负责蓄能电站并网技术方案，运行稳定可靠。具有较强的电力系统运行分析能力。工作认真负责，学习能力强。需要加强智能电网技术应用。"
            },
            {
                "name": "朱储能",
                "age": 31,
                "gender": "女",
                "education": "硕士",
                "school": "新疆大学",
                "major": "新能源科学与工程",
                "department": "抽水蓄能部",
                "position": "储能技术工程师",
                "work_years": 7,
                "political_status": "中共党员",
                "description": "朱储能，女，汉族，1992年8月出生，新疆乌鲁木齐人，硕士研究生学历，新能源科学与工程硕士学位，中级工程师。政治立场坚定，创新意识强。储能技术专业知识扎实，新技术应用能力强。研究储能系统优化运行，提升系统效率。具有较强的技术研发和系统集成能力。工作主动积极，前瞻性强。需要加强产业化应用经验。"
            },

            # 新能源事业部 (4人)
            {
                "name": "风光明",
                "age": 36,
                "gender": "男",
                "education": "博士",
                "school": "北京理工大学",
                "major": "新能源科学与工程",
                "department": "新能源事业部",
                "position": "新能源项目专员",
                "work_years": 11,
                "political_status": "中共党员",
                "description": "风光明，男，汉族，1987年6月出生，宁夏银川人，博士研究生学历，新能源科学与工程博士学位，高级工程师。政治立场坚定，绿色发展理念强。新能源技术专业能力突出，项目开发经验丰富。主持风电、光伏等新能源项目，发电效率领先。具有较强的技术创新和项目管理能力。工作严谨负责，成果转化能力强。需要加强储能系统集成应用。"
            },
            {
                "name": "赵风电",
                "age": 32,
                "gender": "男",
                "education": "硕士",
                "school": "中国海洋大学",
                "major": "风能与动力工程",
                "department": "新能源事业部",
                "position": "风电工程师",
                "work_years": 8,
                "political_status": "中共党员",
                "description": "赵风电，男，汉族，1991年10月出生，山东威海人，硕士研究生学历，风能与动力工程硕士学位，中级工程师。政治觉悟高，环保责任感强。风电技术专业知识扎实，设备选型经验丰富。参与海上风电项目建设，技术难题攻克能力强。具有较强的风能资源分析和发电系统优化能力。工作认真负责，技术钻研精神强。需要加强海上风电专业技术。"
            },
            {
                "name": "钱光伏",
                "age": 28,
                "gender": "女",
                "education": "本科",
                "school": "青海大学",
                "major": "电气工程及其自动化",
                "department": "新能源事业部",
                "position": "光伏工程师",
                "work_years": 4,
                "political_status": "共青团员",
                "description": "钱光伏，女，汉族，1995年4月出生，青海西宁人，本科学历，电气工程学士学位，助理工程师。政治思想积极，创新思维活跃。光伏发电技术基础扎实，系统设计能力强。参与分布式光伏项目建设，发电效果良好。具有较强的光伏系统设计和运维能力。工作主动认真，适应能力强。需要积累更多大型地面电站经验。"
            },
            {
                "name": "孙清洁",
                "age": 39,
                "gender": "女",
                "education": "硕士",
                "school": "西藏大学",
                "major": "环境与能源工程",
                "department": "新能源事业部",
                "position": "清洁能源分析师",
                "work_years": 15,
                "political_status": "中共党员",
                "description": "孙清洁，女，汉族，1984年11月出生，西藏拉萨人，硕士研究生学历，环境与能源工程硕士学位，高级工程师。政治立场坚定，可持续发展理念强。清洁能源技术分析能力强，政策解读准确。研究清洁能源发展趋势，为决策提供技术支撑。具有较强的技术分析和战略规划能力。工作严谨负责，前瞻性强。需要加强国际清洁能源技术跟踪。"
            },

            # 电子商城部 (3人)
            {
                "name": "商小城",
                "age": 30,
                "gender": "女",
                "education": "硕士",
                "school": "深圳大学",
                "major": "电子商务",
                "department": "电子商城部",
                "position": "电子商城运营专员",
                "work_years": 6,
                "political_status": "中共党员",
                "description": "商小城，女，汉族，1993年7月出生，广东深圳人，硕士研究生学历，电子商务硕士学位，中级经济师。政治觉悟高，创新意识强。电子商务专业能力突出，平台运营经验丰富。负责电子商城平台建设运营，用户活跃度持续提升。具有较强的用户运营和数据分析能力。工作认真负责，市场敏感度高。需要加强B2B电商特色功能开发。"
            },
            {
                "name": "李平台",
                "age": 33,
                "gender": "男",
                "education": "本科",
                "school": "福州大学",
                "major": "计算机科学与技术",
                "department": "电子商城部",
                "position": "平台技术工程师",
                "work_years": 9,
                "political_status": "中共党员",
                "description": "李平台，男，汉族，1990年1月出生，福建福州人，本科学历，计算机科学与技术学士学位，中级工程师。政治思想端正，技术能力扎实。电商平台技术开发经验丰富，系统架构设计合理。负责商城平台技术维护升级，系统稳定性高。具有较强的系统开发和运维能力。工作认真负责，学习能力强。需要加强微服务架构技术应用。"
            },
            {
                "name": "王用户",
                "age": 26,
                "gender": "女",
                "education": "本科",
                "school": "海南大学",
                "major": "设计学",
                "department": "电子商城部",
                "position": "用户体验设计师",
                "work_years": 2,
                "political_status": "共青团员",
                "description": "王用户，女，汉族，1997年3月出生，海南海口人，本科学历，设计学学士学位，助理工程师。政治思想积极，设计理念先进。用户体验设计专业能力强，交互设计思维活跃。负责商城平台界面和交互设计，用户满意度高。具有较强的设计创意和用户需求分析能力。工作主动积极，创新能力强。需要积累更多B2B场景设计经验。"
            },

            # 供应链管理部 (3人)
            {
                "name": "链条长",
                "age": 45,
                "gender": "男",
                "education": "硕士",
                "school": "大连海事大学",
                "major": "物流与供应链管理",
                "department": "供应链管理部",
                "position": "供应链管理主管",
                "work_years": 21,
                "political_status": "中共党员",
                "description": "链条长，男，汉族，1978年8月出生，河南洛阳人，硕士研究生学历，物流与供应链管理硕士学位，高级经济师。政治立场坚定，管理经验丰富。供应链管理专业能力突出，全链条管控经验丰富。建立完善的供应链管理体系，供应链效率显著提升。具有较强的战略规划和资源整合能力。工作稳重可靠，业绩突出。需要加强数字化供应链技术应用。"
            },
            {
                "name": "流程优",
                "age": 37,
                "gender": "女",
                "education": "本科",
                "school": "吉林大学",
                "major": "工业工程",
                "department": "供应链管理部",
                "position": "流程优化专员",
                "work_years": 13,
                "political_status": "中共党员",
                "description": "流程优，女，汉族，1986年5月出生，吉林长春人，本科学历，工业工程学士学位，中级工程师。政治觉悟高，流程意识强。流程优化专业能力强，精益管理经验丰富。主导供应链流程梳理优化，运营效率提升25%。具有较强的流程分析和改进能力。工作严谨细致，执行力强。需要加强数字化流程管理技术。"
            },
            {
                "name": "网络通",
                "age": 34,
                "gender": "男",
                "education": "硕士",
                "school": "哈尔滨工业大学",
                "major": "管理科学与工程",
                "department": "供应链管理部",
                "position": "供应网络分析师",
                "work_years": 10,
                "political_status": "中共党员",
                "description": "网络通，男，汉族，1989年9月出生，黑龙江哈尔滨人，硕士研究生学历，管理科学与工程硕士学位，中级工程师。政治思想端正，分析能力强。供应网络分析专业能力强，数据建模技术熟练。构建供应网络优化模型，供应网络布局更加合理。具有较强的数据分析和网络优化能力。工作认真负责，学习能力强。需要加强人工智能算法应用。"
            },

            # 综合管理部 (4人)
            {
                "name": "管综合",
                "age": 48,
                "gender": "男",
                "education": "硕士",
                "school": "贵州大学",
                "major": "公共管理",
                "department": "综合管理部",
                "position": "综合管理主管",
                "work_years": 24,
                "political_status": "中共党员",
                "description": "管综合，男，汉族，1975年6月出生，贵州遵义人，硕士研究生学历，公共管理硕士学位，高级经济师。政治立场坚定，党性修养深厚。综合管理经验丰富，统筹协调能力强。负责公司综合事务管理，各项工作有序推进。具有较强的组织协调和决策执行能力。工作稳重可靠，大局观强。需要加强数字化办公技术应用。"
            },
            {
                "name": "人事通",
                "age": 35,
                "gender": "女",
                "education": "本科",
                "school": "云南大学",
                "major": "人力资源管理",
                "department": "综合管理部",
                "position": "人事专员",
                "work_years": 11,
                "political_status": "中共党员",
                "description": "人事通，女，汉族，1988年10月出生，云南昆明人，本科学历，人力资源管理学士学位，中级经济师。政治觉悟高，服务意识强。人力资源管理专业能力强，员工关系维护良好。负责人员招聘培训管理，人才队伍建设成效显著。具有较强的沟通协调和人员管理能力。工作认真负责，亲和力强。需要加强人才发展规划能力。"
            },
            {
                "name": "财务清",
                "age": 41,
                "gender": "女",
                "education": "硕士",
                "school": "上海财经大学",
                "major": "会计学",
                "department": "综合管理部",
                "position": "财务主管",
                "work_years": 17,
                "political_status": "中共党员",
                "description": "财务清，女，汉族，1982年1月出生，台湾台北人，硕士研究生学历，会计学硕士学位，高级会计师。政治立场坚定，财务意识强。财务管理专业能力突出，成本控制经验丰富。建立完善的财务管理制度，资金使用效率高。具有较强的财务分析和风险控制能力。工作严谨负责，原则性强。需要加强管理会计技术应用。"
            },
            {
                "name": "办公室",
                "age": 28,
                "gender": "男",
                "education": "本科",
                "school": "香港中文大学",
                "major": "汉语言文学",
                "department": "综合管理部",
                "position": "办公室秘书",
                "work_years": 4,
                "political_status": "共青团员",
                "description": "办公室，男，汉族，1995年11月出生，香港特别行政区人，本科学历，汉语言文学学士学位，助理馆员。政治思想积极，文字功底扎实。办公室事务处理能力强，文档管理规范。负责会议组织和文件起草，服务保障到位。具有较强的文字表达和组织协调能力。工作主动认真，服务意识强。需要加强公文写作专业技能。"
            }
        ]
        
        # 创建干部画像对象
        for cadre_data in sample_cadres:
            cadre = CadreProfile(**cadre_data)
            self.cadres.append(cadre)
        
        # 为重点干部添加职业生涯成长轨迹
        self._add_career_trajectories()
        
        # 为所有干部添加亲属关系测试数据
        self._add_family_relations()
    
    def _add_career_trajectories(self):
        """为重点干部添加职业生涯成长轨迹数据"""
        # 张明的职业成长轨迹
        zhang_ming = self.get_cadre_by_name("张明")
        if zhang_ming:
            zhang_ming.career_milestones = [
                {"年份": 2012, "事件": "入职三峡物资招标公司", "岗位": "招投标专员", "意义": "职业生涯起点"},
                {"年份": 2015, "事件": "晋升招投标主管", "岗位": "招投标主管", "意义": "首次担任管理职务"},
                {"年份": 2017, "事件": "获得注册采购师资格", "岗位": "招投标主管", "意义": "专业能力认证"},
                {"年份": 2019, "事件": "主持白鹤滩项目招标", "岗位": "招投标主管", "意义": "承担重大项目责任"},
                {"年份": 2022, "事件": "获得公司管理创新奖", "岗位": "招投标主管", "意义": "创新能力得到认可"}
            ]
            zhang_ming.awards_qualifications = {
                "奖项荣誉": [
                    {"名称": "公司管理创新奖", "级别": "公司级", "获得年份": "2022", "颁发机构": "公司元委会"},
                    {"名称": "优秀管理者称号", "级别": "省部级", "获得年份": "2021", "颁发机构": "省经济和信息化委员会"},
                    {"名称": "突出贡献奖", "级别": "部门级", "获得年份": "2020", "颁发机构": "采购部"}
                ],
                "职称资质": [
                    {"名称": "采购师高级认证", "级别": "高级", "获得年份": "2020", "颁发机构": "中国物流采购联合会"},
                    {"名称": "PMP项目管理认证", "级别": "国际认证", "获得年份": "2021", "颁发机构": "PMI"},
                    {"名称": "高级经济师", "级别": "高级", "获得年份": "2019", "颁发机构": "人力资源社会保障部"}
                ]
            }
            zhang_ming.performance_history = [
                {"年份": 2020, "考核等级": "优秀", "关键成果": "完成招标项目50项", "改进建议": "加强国际业务"},
                {"年份": 2021, "考核等级": "优秀", "关键成果": "提升采购效率35%", "改进建议": "强化数字化应用"},
                {"年份": 2022, "考核等级": "优秀", "关键成果": "建立风险预警机制", "改进建议": "扩展跨部门协作"},
                {"年份": 2023, "考核等级": "优秀", "关键成果": "团队获先进集体", "改进建议": "提升战略思维"}
            ]
            zhang_ming.training_records = [
                {"年份": 2020, "培训内容": "采购师高级认证", "培训机构": "中国物流采购联合会", "学时": 80},
                {"年份": 2021, "培训内容": "项目管理PMP认证", "培训机构": "PMI", "学时": 120},
                {"年份": 2022, "培训内容": "数字化采购专题", "培训机构": "清华大学", "学时": 40},
                {"年份": 2023, "培训内容": "领导力提升训练", "培训机构": "三峡大学", "学时": 60}
            ]
        
        # 李华的职业成长轨迹
        li_hua = self.get_cadre_by_name("李华")
        if li_hua:
            li_hua.career_milestones = [
                {"年份": 2008, "事件": "研究生毕业入职", "岗位": "物资管理员", "意义": "专业对口就业"},
                {"年份": 2012, "事件": "晋升物资管理专员", "岗位": "物资管理专员", "意义": "专业能力认可"},
                {"年份": 2015, "事件": "获得高级经济师", "岗位": "物资管理专员", "意义": "职业资格提升"},
                {"年份": 2019, "事件": "主导库存优化项目", "岗位": "物资管理专员", "意义": "管理创新突破"},
                {"年份": 2021, "事件": "获得科技进步奖", "岗位": "物资管理专员", "意义": "技术创新成果"}
            ]
            li_hua.awards_qualifications = {
                "奖项荣誉": [
                    {"名称": "科技进步奖", "级别": "公司级", "获得年份": "2021", "颁发机构": "公司科技委"},
                    {"名称": "优秀员工", "级别": "部门级", "获得年份": "2020", "颁发机构": "物资部"},
                    {"名称": "技术创新奖", "级别": "部门级", "获得年份": "2019", "颁发机构": "物资部"}
                ],
                "职称资质": [
                    {"名称": "高级经济师", "级别": "高级", "获得年份": "2015", "颁发机构": "人力资源社会保障部"},
                    {"名称": "供应链管理师", "级别": "中级", "获得年份": "2018", "颁发机构": "中国物流与采购联合会"},
                    {"名称": "物流师认证", "级别": "中级", "获得年份": "2017", "颁发机构": "中国物流学会"}
                ]
            }
            li_hua.performance_history = [
                {"年份": 2020, "考核等级": "优秀", "关键成果": "降低库存成本20%", "改进建议": "加强技术创新"},
                {"年份": 2021, "考核等级": "优秀", "关键成果": "提高周转率40%", "改进建议": "拓展国际视野"},
                {"年份": 2022, "考核等级": "优秀", "关键成果": "库存准确率99.8%", "改进建议": "强化团队建设"},
                {"年份": 2023, "考核等级": "优秀", "关键成果": "信息化项目成功", "改进建议": "提升创新能力"}
            ]
        
        # 王强的职业成长轨迹
        wang_qiang = self.get_cadre_by_name("王强")
        if wang_qiang:
            wang_qiang.career_milestones = [
                {"年份": 2013, "事件": "博士毕业入职", "岗位": "采购专员", "意义": "高学历人才引进"},
                {"年份": 2016, "事件": "晋升采购主管", "岗位": "采购主管", "意义": "快速成长典型"},
                {"年份": 2018, "事件": "获得国际采购认证", "岗位": "采购主管", "意义": "国际化能力"},
                {"年份": 2020, "事件": "主导百亿采购项目", "岗位": "采购主管", "意义": "重大项目历练"},
                {"年份": 2022, "事件": "获得科技进步一等奖", "岗位": "采购主管", "意义": "创新成果突出"}
            ]
            wang_qiang.awards_qualifications = {
                "奖项荣誉": [
                    {"名称": "科技进步一等奖", "级别": "省部级", "获得年份": "2022", "颁发机构": "省科技厅"},
                    {"名称": "优秀青年专家", "级别": "公司级", "获得年份": "2021", "颁发机构": "公司人才委员会"},
                    {"名称": "采购管理先进个人", "级别": "部门级", "获得年份": "2020", "颁发机构": "采购部"}
                ],
                "职称资质": [
                    {"名称": "国际采购认证CIPS", "级别": "国际认证", "获得年份": "2018", "颁发机构": "英国皇家采购与供应学会"},
                    {"名称": "高级经济师", "级别": "高级", "获得年份": "2020", "颁发机构": "人力资源社会保障部"},
                    {"名称": "注册采购师CPO", "级别": "国家级", "获得年份": "2019", "颁发机构": "中国采购与物流联合会"}
                ]
            }
            wang_qiang.performance_history = [
                {"年份": 2020, "考核等级": "优秀", "关键成果": "节约成本2.5亿", "改进建议": "加强国际经验"},
                {"年份": 2021, "考核等级": "优秀", "关键成果": "建立全球供应网", "改进建议": "深化技术应用"},
                {"年份": 2022, "考核等级": "优秀", "关键成果": "制度建设突出", "改进建议": "提升团队能力"},
                {"年份": 2023, "考核等级": "优秀", "关键成果": "培训人员500人", "改进建议": "拓展新兴领域"}
            ]
    
    def _add_family_relations(self):
        """为所有干部添加亲属关系测试数据"""
        import random
        
        # 常用中文姓氏
        surnames = ["张", "王", "李", "赵", "刘", "陈", "杨", "黄", "周", "吴", "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗", "梁", "宋", "郑", "谢", "韩", "唐", "冯", "于", "董", "萧", "程", "曹", "袁", "邓", "许", "傅", "沈", "曾", "彭", "吕", "苏", "卢", "蒋", "蔡", "贾", "丁", "魏", "薛", "叶", "阎"]
        
        # 常用名字
        male_names = ["建国", "志强", "志华", "国强", "建华", "国华", "志明", "建军", "志刚", "国民", "建设", "志勇", "国庆", "建平", "志伟", "建东", "志峰", "国兴", "建中", "志鹏", "国栋", "建伟", "志超", "国良", "建斌", "志龙", "国平", "建新", "志飞", "国辉"]
        female_names = ["丽华", "雅静", "美玲", "晓敏", "丽娟", "小芳", "雪梅", "小燕", "丽萍", "晓燕", "美芳", "雅琪", "小丽", "丽娜", "美玉", "晓霞", "雅雯", "小红", "丽英", "美丽", "晓梅", "雅婷", "小玉", "丽芳", "美华", "晓雨", "雅娜", "小慧", "丽君", "美娜"]
        
        # 详细的工作单位和职位信息
        work_info = {
            # 国有企业
            "三峡集团有限公司": ["人力资源部经理", "财务部主管", "工程管理部总监", "安全环保部副主任", "党委办公室主任", "纪检监察部专员", "技术研发中心主任", "市场营销部经理", "法务部专员", "投资发展部副经理"],
            "中国石油化工集团": ["炼化事业部经理", "销售公司区域经理", "安全环保部主管", "技术研发院高级工程师", "人力资源部招聘主管", "财务部会计师", "物资采购部采购员", "质量管理部质检员", "设备维护部技师", "党群工作部干事"],
            "国家电网公司": ["变电运维班长", "配电抢修队长", "电力调度中心调度员", "营销服务部客户经理", "人力资源部培训师", "财务部预算分析师", "信息通信部网络工程师", "安全监察部安全员", "物资部仓储管理员", "党建部宣传干事"],
            "中国建筑集团": ["项目经理", "造价工程师", "安全管理员", "质量检查员", "技术负责人", "材料采购员", "财务会计", "人力资源专员", "法务合规专员", "党务工作者"],
            "中国移动通信": ["网络优化工程师", "客户服务经理", "市场营销专员", "技术支持工程师", "财务分析师", "人力资源招聘专员", "企业客户经理", "网络维护技师", "数据分析师", "党建工作专员"],
            "中国建设银行宜昌分行": ["客户经理", "信贷审批员", "柜面业务员", "风险管理专员", "财富管理顾问", "运营管理员", "科技部系统管理员", "合规检查员", "人力资源专员", "党建工作干事"],
            "中国工商银行": ["个人银行部客户经理", "公司业务部信贷员", "风险管理部风控专员", "运营管理部主管", "科技部软件工程师", "人力资源部培训师", "审计部内审员", "合规部合规官", "财务部会计师", "办公室文秘"],
            
            # 医疗卫生事业单位
            "宜昌市第一人民医院": ["心内科主任医师", "神经外科副主任医师", "急诊科主治医师", "护理部护士长", "医务处副处长", "财务科会计师", "人事科干事", "药剂科药师", "检验科技师", "放射科技师"],
            "宜昌市中心医院": ["骨科主任医师", "妇产科副主任医师", "儿科主治医师", "ICU护士长", "医保办主任", "设备科工程师", "信息科系统管理员", "总务科后勤管理员", "党办干事", "工会主席"],
            "湖北省中医院": ["中医内科主任医师", "针灸科副主任医师", "中药房主管药师", "治未病中心主任", "护理部总护士长", "医务处处长", "科教处副处长", "人事处干事", "财务处会计", "党委办公室主任"],
            
            # 教育事业单位
            "华中科技大学": ["计算机学院教授", "机械学院副教授", "电气学院讲师", "管理学院助理教授", "人事处处长", "教务处副处长", "科研院院长", "学生处辅导员", "财务处会计师", "后勤集团经理"],
            "宜昌市第一中学": ["高中数学高级教师", "语文教研组长", "英语备课组长", "物理实验员", "德育处主任", "教务处副主任", "总务处主任", "团委书记", "工会主席", "党支部书记"],
            "三峡大学": ["水利工程学院教授", "电气工程学院副教授", "经济管理学院讲师", "人事处处长", "教务处副处长", "学生工作处处长", "后勤服务集团总经理", "图书馆馆长", "科技处处长", "财务处处长"],
            
            # 政府机关
            "宜昌市发展和改革委员会": ["综合规划科科长", "投资管理科副科长", "价格监测中心主任", "产业发展科干事", "办公室主任", "人事教育科科长", "法规科科长", "党组织书记", "纪检组长", "工会主席"],
            "宜昌市财政局": ["预算管理科科长", "国库支付中心主任", "政府采购办主任", "税政科副科长", "会计管理科科长", "监督检查科科长", "人事科科长", "办公室主任", "党委委员", "工会主席"],
            "宜昌市人力资源和社会保障局": ["就业促进科科长", "社会保障科副科长", "人事考试院院长", "劳动监察支队队长", "工伤保险科科长", "养老保险科科长", "人才交流中心主任", "职业技能鉴定中心主任", "办公室主任", "党组成员"],
            
            # 民营企业
            "宜昌东方物流有限公司": ["运营总监", "仓储部经理", "运输调度主管", "客户服务经理", "财务部会计师", "人事行政专员", "信息技术部主管", "安全管理员", "质量控制员", "市场开发经理"],
            "湖北三峡科技有限公司": ["研发部技术总监", "产品经理", "软件开发工程师", "测试工程师", "销售经理", "市场推广专员", "人力资源主管", "财务经理", "行政助理", "项目经理"],
            "宜昌建筑工程有限公司": ["项目总监", "工程部经理", "造价部主管", "安全管理员", "质量检查员", "技术负责人", "财务部会计", "人事专员", "材料采购员", "办公室主任"],
            
            # 退休情况
            "已退休": ["原三峡集团工程师（已退休）", "原宜昌市政府办公室主任（已退休）", "原市第一医院主任医师（已退休）", "原华中科技大学教授（已退休）", "原中国建设银行支行长（已退休）", "原市教育局局长（已退休）", "原国家电网高级工程师（已退休）", "原宜昌钢铁厂工程师（已退休）", "原市人民医院护士长（已退休）", "原宜昌一中高级教师（已退休）"],
            
            # 学生情况
            "在读学生": ["小学生", "初中生", "高中生"],
            "在读大学生": ["华中科技大学本科生", "三峡大学研究生", "武汉大学博士生", "中南财经政法大学本科生", "湖北工业大学硕士生"],
            "职业技术学院学生": ["宜昌职业技术学院学生", "三峡电力职业学院学生", "湖北三峡职业技术学院学生"]
        }
        
        for cadre in self.cadres:
            # 根据干部年龄合理设置父母年龄和工作状态
            father_age = cadre.age + random.randint(25, 35)
            mother_age = cadre.age + random.randint(23, 33)
            
            # 父亲信息
            father_surname = random.choice(surnames)
            father_name_part = random.choice(male_names)
            father_name = father_surname + father_name_part
            
            # 根据父亲年龄决定工作状态
            if father_age >= 60:
                father_work_unit = "已退休"
                father_job_position = random.choice(work_info["已退休"])
            else:
                # 选择一个非退休的工作单位
                available_units = [unit for unit in work_info.keys() if unit not in ["已退休", "在读学生", "在读大学生", "职业技术学院学生"]]
                father_work_unit = random.choice(available_units)
                father_job_position = random.choice(work_info[father_work_unit])
            
            # 母亲信息（通常与父亲同姓，但也可能保持娘家姓）
            if random.random() > 0.3:  # 70%概率与父亲同姓
                mother_surname = father_surname
            else:
                mother_surname = random.choice(surnames)
            mother_name_part = random.choice(female_names)
            mother_name = mother_surname + mother_name_part
            
            # 根据母亲年龄决定工作状态
            if mother_age >= 55:  # 女性退休年龄通常较早
                mother_work_unit = "已退休"
                mother_job_position = random.choice(work_info["已退休"])
            else:
                # 选择一个非退休的工作单位
                available_units = [unit for unit in work_info.keys() if unit not in ["已退休", "在读学生", "在读大学生", "职业技术学院学生"]]
                mother_work_unit = random.choice(available_units)
                mother_job_position = random.choice(work_info[mother_work_unit])
            
            # 配偶信息（根据年龄决定是否有配偶）
            spouse_name = ""
            spouse_work_unit = ""
            spouse_job_position = ""
            if cadre.age >= 25:  # 25岁以上可能有配偶
                has_spouse_probability = 0.85 if cadre.age >= 30 else 0.6
                if random.random() < has_spouse_probability:
                    spouse_surname = random.choice(surnames)
                    if cadre.gender == "男":
                        spouse_name_part = random.choice(female_names)
                    else:
                        spouse_name_part = random.choice(male_names)
                    spouse_name = spouse_surname + spouse_name_part
                    
                    # 为配偶选择工作单位和职位
                    available_units = [unit for unit in work_info.keys() if unit not in ["已退休", "在读学生", "在读大学生", "职业技术学院学生"]]
                    spouse_work_unit = random.choice(available_units)
                    spouse_job_position = random.choice(work_info[spouse_work_unit])
            
            # 子女信息（根据年龄决定子女数量）
            children = []
            if cadre.age >= 28:  # 28岁以上可能有子女
                if cadre.age < 35:
                    children_count = random.choices([0, 1, 2], weights=[40, 50, 10])[0]
                elif cadre.age < 45:
                    children_count = random.choices([0, 1, 2, 3], weights=[10, 40, 40, 10])[0]
                else:
                    children_count = random.choices([1, 2, 3], weights=[30, 50, 20])[0]
                
                for i in range(children_count):
                    child_age = random.randint(1, min(cadre.age - 20, 30))
                    child_surname = cadre.name[0]  # 子女通常随父姓
                    
                    if random.random() > 0.5:
                        child_name_part = random.choice(male_names)
                        child_gender = "男"
                    else:
                        child_name_part = random.choice(female_names)
                        child_gender = "女"
                    
                    child_name = child_surname + child_name_part
                    
                    # 根据子女年龄决定工作状态
                    if child_age < 18:
                        child_work_unit = "在读学生"
                        child_job_position = random.choice(work_info["在读学生"])
                    elif child_age < 22:
                        if random.random() < 0.7:  # 70%概率还在读书
                            child_work_unit = random.choice(["在读大学生", "职业技术学院学生"])
                            child_job_position = random.choice(work_info[child_work_unit])
                        else:  # 30%概率已参加工作
                            available_units = [unit for unit in work_info.keys() if unit not in ["已退休", "在读学生", "在读大学生", "职业技术学院学生"]]
                            child_work_unit = random.choice(available_units)
                            child_job_position = random.choice(work_info[child_work_unit])
                    else:
                        # 22岁以上基本都参加工作了
                        available_units = [unit for unit in work_info.keys() if unit not in ["已退休", "在读学生", "在读大学生", "职业技术学院学生"]]
                        child_work_unit = random.choice(available_units)
                        child_job_position = random.choice(work_info[child_work_unit])
                    
                    children.append({
                        "name": child_name,
                        "work_unit": child_work_unit,
                        "job_position": child_job_position
                    })
            
            # 设置亲属关系信息
            cadre.family_relations = {
                "father": {
                    "name": father_name,
                    "work_unit": father_work_unit,
                    "job_position": father_job_position
                },
                "mother": {
                    "name": mother_name,
                    "work_unit": mother_work_unit,
                    "job_position": mother_job_position
                },
                "spouse": {
                    "name": spouse_name,
                    "work_unit": spouse_work_unit,
                    "job_position": spouse_job_position
                },
                "children": children
            }
    
    def get_all_cadres(self) -> List[CadreProfile]:
        """获取所有干部"""
        return self.cadres
    
    def get_cadre_by_name(self, name: str) -> Optional[CadreProfile]:
        """根据姓名获取干部"""
        for cadre in self.cadres:
            if cadre.name == name:
                return cadre
        return None
    
    def add_cadre(self, cadre: CadreProfile):
        """添加干部"""
        self.cadres.append(cadre)
    
    def update_cadre(self, cadre: CadreProfile):
        """更新干部信息"""
        for i, existing_cadre in enumerate(self.cadres):
            if existing_cadre.name == cadre.name:
                self.cadres[i] = cadre
                break

# ===== 全局变量初始化 =====
@st.cache_resource
def init_components():
    """初始化组件"""
    data_manager = DataManager()
    analyzer = PersonalMaterialAnalyzer()
    river_model = RiverCompetencyModel()
    matching_engine = PositionMatchingEngine()
    
    # 分析所有干部的个人材料
    for cadre in data_manager.get_all_cadres():
        analyzer.analyze_personal_material(cadre)
        cadre.river_competency = river_model.calculate_river_competency(cadre)
        cadre.development_dimensions = river_model.calculate_development_dimensions(cadre)
    
    return data_manager, analyzer, river_model, matching_engine

# ===== 主界面 =====
def main():
    """主函数"""
    # 初始化组件
    data_manager, analyzer, river_model, matching_engine = init_components()
    
    # 主标题
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ 三峡物资招标公司干部画像管理系统</h1>
        <p>三峡物资招标公司数字化中心 | 智慧识才解决方案</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏导航
    st.sidebar.title("🎯 功能导航")
    
    menu_options = [
        "📊 系统概览",
        "👤 干部画像分析", 
        "🌊 江河胜任图模型",
        "🎯 智能人岗匹配",
        "🕸️ 干部关系网络",
        "📈 数据可视化分析",
        "🔍 人才发展建议",
        "➕ 添加干部信息"
    ]
    
    selected_menu = st.sidebar.radio("选择功能", menu_options)
    
    # 根据选择显示相应页面
    if selected_menu == "📊 系统概览":
        show_system_overview(data_manager)
    elif selected_menu == "👤 干部画像分析":
        show_five_dimension_analysis(data_manager, analyzer)
    elif selected_menu == "🌊 江河胜任图模型":
        show_river_competency_model(data_manager, river_model)
    elif selected_menu == "🎯 智能人岗匹配":
        show_position_matching(data_manager, matching_engine)
    elif selected_menu == "🕸️ 干部关系网络":
        show_cadre_relationship_network(data_manager)
    elif selected_menu == "📈 数据可视化分析":
        show_data_visualization(data_manager)
    elif selected_menu == "🔍 人才发展建议":
        show_development_suggestions(data_manager, river_model)
    elif selected_menu == "➕ 添加干部信息":
        show_add_cadre_form(data_manager, analyzer, river_model)

def show_cadre_relationship_network(data_manager: DataManager):
    """显示干部关系网络分析"""
    st.header("🕸️ 干部关系网络")
    st.markdown("*干部关系网络与廉洁合规风险预防*")
    
    cadres = data_manager.get_all_cadres()
    cadre_names = [cadre.name for cadre in cadres]
    
    # 选择功能模式
    analysis_mode = st.selectbox("选择分析模式", [
        "🏢 个人工作关系网络",
        "👨‍👩‍👧‍👦 个人亲属关系网络",
        "⚠️ 廉洁风险评估", 
        "📊 整体关系分析"
    ])
    
    if analysis_mode == "🏢 个人工作关系网络":
        show_personal_work_network(data_manager, cadre_names)
    
    elif analysis_mode == "👨‍👩‍👧‍👦 个人亲属关系网络":
        show_personal_family_network(data_manager, cadre_names)
    
    elif analysis_mode == "⚠️ 廉洁风险评估":
        show_integrity_risk_assessment(data_manager, cadre_names)
    
    elif analysis_mode == "📊 整体关系分析":
        show_overall_relationship_analysis(data_manager, cadre_names)

def show_system_overview(data_manager: DataManager):
    """显示系统概览"""
    st.header("📊 系统概览")
    
    cadres = data_manager.get_all_cadres()
    
    # 统计数据
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👥 干部总数</h3>
            <h1 style="color: #667eea;">{}</h1>
        </div>
        """.format(len(cadres)), unsafe_allow_html=True)
    
    with col2:
        avg_age = sum(cadre.age for cadre in cadres) / len(cadres)
        st.markdown("""
        <div class="metric-card">
            <h3>📅 平均年龄</h3>
            <h1 style="color: #667eea;">{:.1f}</h1>
        </div>
        """.format(avg_age), unsafe_allow_html=True)
    
    with col3:
        avg_work_years = sum(cadre.work_years for cadre in cadres) / len(cadres)
        st.markdown("""
        <div class="metric-card">
            <h3>💼 平均工作年限</h3>
            <h1 style="color: #667eea;">{:.1f}</h1>
        </div>
        """.format(avg_work_years), unsafe_allow_html=True)
    
    with col4:
        high_performers = sum(1 for cadre in cadres if cadre.performance >= 80)
        st.markdown("""
        <div class="metric-card">
            <h3>⭐ 优秀人员数</h3>
            <h1 style="color: #667eea;">{}</h1>
        </div>
        """.format(high_performers), unsafe_allow_html=True)
    
    # 部门分布
    st.subheader("🏢 部门分布")
    departments = {}
    for cadre in cadres:
        departments[cadre.department] = departments.get(cadre.department, 0) + 1
    
    fig_dept = px.pie(
        values=list(departments.values()),
        names=list(departments.keys()),
        title="部门人员分布",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_dept, use_container_width=True)
    
    # 教育背景分布
    st.subheader("🎓 教育背景分布")
    education_dist = {}
    for cadre in cadres:
        education_dist[cadre.education] = education_dist.get(cadre.education, 0) + 1
    
    fig_edu = px.bar(
        x=list(education_dist.keys()),
        y=list(education_dist.values()),
        title="教育背景分布",
        color=list(education_dist.values()),
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_edu, use_container_width=True)
    
    # 最近更新
    st.subheader("📈 系统特色")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card">
            <h4>🔍 智能分析</h4>
            <p>• 基于NLP的个人材料分析</p>
            <p>• 自动提取五维画像标签</p>
            <p>• 智能情感分析评估</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <h4>🌊 江河胜任图</h4>
            <p>• 三峡物资招投标特色模型</p>
            <p>• 四大核心能力维度</p>
            <p>• 专业岗位匹配分析</p>
        </div>
        """, unsafe_allow_html=True)

def show_five_dimension_analysis(data_manager: DataManager, analyzer: PersonalMaterialAnalyzer):
    """显示干部画像分析"""
    st.header("👤 干部画像分析")
    st.markdown("*基于NLP智能分析的个人材料提取*")
    
    cadres = data_manager.get_all_cadres()
    cadre_names = [cadre.name for cadre in cadres]
    
    selected_cadre_name = st.selectbox("选择干部", cadre_names)
    cadre = data_manager.get_cadre_by_name(selected_cadre_name)
    
    if cadre:
        # 干部基础信息卡片
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;
                   box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
            <h3 style="margin: 0 0 15px 0; color: white;">👤 干部基础信息</h3>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">姓名</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.name}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">部门</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.department}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">职位</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.position}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">年龄</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.age}岁</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">性别</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.gender}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">学历</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.education}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">毕业院校</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.school if cadre.school else '未录入'}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">专业</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.major if cadre.major else '未录入'}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">政治身份</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.political_status}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; opacity: 0.8;">工作年限</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin-top: 5px;">{cadre.work_years}年</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 五维画像分析
        st.subheader("📊 五维画像分析")
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            # 五维画像雷达图
            dimensions = ["素质基础", "胜任能力", "工作绩效", "自画像", "声誉得分"]
            scores = [cadre.quality_foundation, cadre.competency, cadre.performance, 
                     cadre.self_portrait, cadre.reputation_score]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions,
                fill='toself',
                name=cadre.name,
                line_color='rgb(102, 126, 234)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title=f"{cadre.name} 五维画像雷达图",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 维度得分 - 紧凑布局
            st.markdown("**维度得分**")
            for i, (dimension, score) in enumerate(zip(dimensions, scores)):
                color = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"][i]
                st.markdown(f"""
                <div style="background: white; padding: 10px; border-radius: 8px; margin-bottom: 8px;
                           border-left: 4px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; font-size: 0.9em;">{dimension}</span>
                        <span style="color: {color}; font-weight: bold; font-size: 1.1em;">{score:.1f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with col3:
            # 综合评价
            avg_score = sum(scores) / len(scores)
            if avg_score >= 85:
                level = "优秀"
                level_color = "#27AE60"
                level_icon = "🌟"
            elif avg_score >= 75:
                level = "良好"
                level_color = "#3498DB"
                level_icon = "👍"
            elif avg_score >= 65:
                level = "合格"
                level_color = "#F39C12"
                level_icon = "✅"
            else:
                level = "待提升"
                level_color = "#E74C3C"
                level_icon = "📈"
                
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {level_color}20 0%, {level_color}10 100%);
                       border: 2px solid {level_color}; border-radius: 12px; padding: 15px; text-align: center;">
                <div style="font-size: 2em; margin-bottom: 10px;">{level_icon}</div>
                <div style="font-size: 1.2em; font-weight: bold; color: {level_color}; margin-bottom: 5px;">{level}</div>
                <div style="font-size: 1.5em; font-weight: bold; color: {level_color};">{avg_score:.1f}</div>
                <div style="font-size: 0.8em; color: #666; margin-top: 5px;">综合评分</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 排名信息
            all_cadres = data_manager.get_all_cadres()
            all_scores = []
            for c in all_cadres:
                c_scores = [c.quality_foundation, c.competency, c.performance, c.self_portrait, c.reputation_score]
                all_scores.append((c.name, sum(c_scores) / len(c_scores)))
            
            all_scores.sort(key=lambda x: x[1], reverse=True)
            current_rank = next((i+1 for i, (name, _) in enumerate(all_scores) if name == cadre.name), 0)
            
            st.markdown(f"""
            <div style="background: #f8f9fa; border-radius: 8px; padding: 10px; margin-top: 10px; text-align: center;">
                <div style="font-size: 0.9em; color: #666;">综合排名</div>
                <div style="font-size: 1.2em; font-weight: bold; color: #333;">{current_rank}/{len(all_cadres)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 提取的关键词标签 - 优化视觉效果
        st.subheader("🏷️ 智能提取标签")
        
        if cadre.extracted_tags:
            # 使用选项卡布局展示不同维度
            dimension_names = [dim for dim in cadre.extracted_tags.keys() 
                             if any(tags for tags in cadre.extracted_tags[dim].values())]
            
            if dimension_names:
                tab_cols = st.tabs(dimension_names)
                
                # 定义维度颜色
                dimension_colors = {
                    "素质基础": "#FF6B6B",
                    "胜任能力": "#4ECDC4", 
                    "工作绩效": "#45B7D1",
                    "自画像": "#96CEB4",
                    "声誉得分": "#E74C3C"
                }
                
                for i, dimension in enumerate(dimension_names):
                    with tab_cols[i]:
                        categories = cadre.extracted_tags[dimension]
                        color = dimension_colors.get(dimension, "#667eea")
                        
                        # 统计该维度的标签总数
                        total_tags = sum(len(tags) for tags in categories.values())
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                                    border-left: 4px solid {color}; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <h4 style="color: {color}; margin: 0;">📊 {dimension}</h4>
                            <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #666;">
                                共提取到 <strong>{total_tags}</strong> 个关键标签
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 按类别展示标签，使用卡片布局
                        for category, tags in categories.items():
                            if tags:
                                # 计算标签重要性（基于数量给予权重）
                                tag_importance = min(len(tags) / 3, 1.0)  # 最多3个标签为满分
                                importance_color = color if tag_importance > 0.6 else "#95a5a6"
                                
                                st.markdown(f"""
                                <div style="background: white; border-radius: 8px; padding: 12px; 
                                           margin-bottom: 10px; border: 1px solid #e0e0e0;
                                           box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                        <div style="width: 8px; height: 8px; border-radius: 50%; 
                                                   background: {importance_color}; margin-right: 8px;"></div>
                                        <strong style="color: #333; font-size: 0.95em;">{category}</strong>
                                        <span style="margin-left: auto; background: {color}20; 
                                                    color: {color}; padding: 2px 8px; border-radius: 12px; 
                                                    font-size: 0.8em;">{len(tags)}个</span>
                                    </div>
                                    <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                                """, unsafe_allow_html=True)
                                
                                for tag in tags:
                                    st.markdown(f"""
                                    <span style="background: {color}15; color: {color}; 
                                                padding: 4px 8px; border-radius: 15px; font-size: 0.85em;
                                                border: 1px solid {color}30; display: inline-block;">
                                        {tag}
                                    </span>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("</div></div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; 
                                           margin-bottom: 10px; border: 1px solid #e9ecef;">
                                    <div style="display: flex; align-items: center;">
                                        <div style="width: 8px; height: 8px; border-radius: 50%; 
                                                   background: #dee2e6; margin-right: 8px;"></div>
                                        <span style="color: #6c757d; font-size: 0.9em;">{category}</span>
                                        <span style="margin-left: auto; color: #6c757d; font-size: 0.8em;">暂无标签</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # 添加标签云视图
                st.subheader("☁️ 标签云")
                all_tags = []
                for dimension, categories in cadre.extracted_tags.items():
                    for category, tags in categories.items():
                        for tag in tags:
                            all_tags.append({"tag": tag, "dimension": dimension})
                
                if all_tags:
                    # 按维度分组显示标签云
                    cols = st.columns(len(dimension_colors))
                    for i, (dimension, color) in enumerate(dimension_colors.items()):
                        if i < len(cols):
                            with cols[i]:
                                dimension_tags = [item["tag"] for item in all_tags if item["dimension"] == dimension]
                                if dimension_tags:
                                    st.markdown(f"**{dimension}**")
                                    tags_html = " ".join([
                                        f'<span style="background: {color}20; color: {color}; '
                                        f'padding: 3px 8px; border-radius: 12px; font-size: 0.8em; '
                                        f'margin: 2px; display: inline-block;">{tag}</span>'
                                        for tag in dimension_tags[:5]  # 最多显示5个标签
                                    ])
                                    st.markdown(f'<div style="line-height: 2;">{tags_html}</div>', 
                                              unsafe_allow_html=True)
                else:
                    st.info("暂无提取到的标签")
            else:
                st.info("该干部的个人材料中暂未识别到相关标签")
        else:
            st.info("正在分析个人材料，请稍候...")
        
        # 职业生涯成长轨迹
        st.subheader("📈 职业生涯成长轨迹")
        show_career_trajectory(cadre, data_manager)
        
        # 个人材料原文
        st.subheader("📄 个人材料原文")
        with st.expander("查看详细材料"):
            st.text_area("个人描述", cadre.description, height=200, disabled=True)

def show_career_trajectory(cadre: CadreProfile, data_manager: DataManager):
    """显示职业生涯成长轨迹"""
    if not cadre.career_milestones:
        st.info("该干部的职业生涯数据暂未完善，正在补充中...")
        return
    
    # 职业轨迹概览
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%); 
                   color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 1.5em; margin-bottom: 5px;">🗺️</div>
            <div style="font-size: 1.2em; font-weight: bold;">{}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">工作年限</div>
        </div>
        """.format(f"{cadre.work_years}年"), unsafe_allow_html=True)
    
    with col2:
        milestone_count = len(cadre.career_milestones)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4ECDC4 0%, #6EE7E7 100%); 
                   color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 1.5em; margin-bottom: 5px;">🎆</div>
            <div style="font-size: 1.2em; font-weight: bold;">{}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">重要里程碑</div>
        </div>
        """.format(milestone_count), unsafe_allow_html=True)
    
    with col3:
        training_count = len(cadre.training_records) if cadre.training_records else 0
        st.markdown("""
        <div style="background: linear-gradient(135deg, #45B7D1 0%, #67C3E5 100%); 
                   color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 1.5em; margin-bottom: 5px;">🎓</div>
            <div style="font-size: 1.2em; font-weight: bold;">{}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">培训认证</div>
        </div>
        """.format(training_count), unsafe_allow_html=True)
    
    with col4:
        avg_performance = 0
        if cadre.performance_history:
            performance_scores = []
            for perf in cadre.performance_history:
                if perf.get('考核等级') == '优秀':
                    performance_scores.append(95)
                elif perf.get('考核等级') == '良好':
                    performance_scores.append(85)
                elif perf.get('考核等级') == '合格':
                    performance_scores.append(75)
                else:
                    performance_scores.append(65)
            avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #96CEB4 0%, #B8E6C1 100%); 
                   color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 1.5em; margin-bottom: 5px;">🏆</div>
            <div style="font-size: 1.2em; font-weight: bold;">{:.1f}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">平均绩效</div>
        </div>
        """.format(avg_performance), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 职业发展时间轴
    tab1, tab2, tab3, tab4 = st.tabs(["📈 职业轨迹", "🏆 奖项资质", "🎓 培训记录", "📊 绩效表现"])
    
    with tab1:
        # 职业里程碑时间轴
        st.markdown("**职业发展时间轴**")
        
        # 获取并排序里程碑
        sorted_milestones = sorted(cadre.career_milestones, key=lambda x: x.get('年份', 0))
        
        for i, milestone in enumerate(sorted_milestones):
            year = milestone.get('年份', '')
            event = milestone.get('事件', '')
            position = milestone.get('岗位', '')
            significance = milestone.get('意义', '')
            
            # 确定事件类型和颜色
            if '入职' in event or '加入' in event:
                event_type = '入职'
                color = '#4ECDC4'
                icon = '💼'
            elif '晋升' in event or '提升' in event or '任命' in event:
                event_type = '晋升'
                color = '#45B7D1'
                icon = '🚀'
            elif '项目' in event or '成果' in event or '奖励' in event:
                event_type = '成就'
                color = '#FECA57'
                icon = '🏆'
            elif '转岗' in event or '调动' in event:
                event_type = '转岗'
                color = '#96CEB4'
                icon = '🔄'
            else:
                event_type = '发展'
                color = '#667eea'
                icon = '🌱'
            
            # 时间轴节点
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: 20px;">
                <div style="flex-shrink: 0; margin-right: 15px;">
                    <div style="background: {color}; color: white; width: 80px; height: 80px; 
                               border-radius: 50%; display: flex; align-items: center; justify-content: center;
                               font-size: 1.5em; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        {icon}
                    </div>
                    <div style="text-align: center; margin-top: 5px; font-size: 0.8em; font-weight: bold; color: {color};">
                        {year}
                    </div>
                </div>
                <div style="flex: 1; background: white; padding: 15px; border-radius: 10px; 
                           box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #333;">{event}</h4>
                        <span style="background: {color}20; color: {color}; padding: 4px 8px; border-radius: 12px; 
                                    font-size: 0.8em; font-weight: bold;">{event_type}</span>
                    </div>
                    <div style="color: #666; font-size: 0.9em; margin-bottom: 5px;">
                        <strong>岗位：</strong>{position}
                    </div>
                    <div style="color: #555; font-size: 0.85em; line-height: 1.4;">
                        {significance}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 添加连接线（除了最后一个）
            if i < len(sorted_milestones) - 1:
                st.markdown("""
                <div style="margin-left: 40px; width: 2px; height: 20px; background: #ddd; margin-bottom: 10px;"></div>
                """, unsafe_allow_html=True)
    
    with tab2:
        # 奖项、职称、资质等
        st.markdown("**奖项、职称、资质认证**")
        
        if cadre.awards_qualifications:
            award_categories = cadre.awards_qualifications
            
            for category, items in award_categories.items():
                if items:
                    st.markdown(f"**{category}**")
                    
                    # 创建奖项资质展示卡片
                    for item in items:
                        item_name = item.get('名称', '')
                        level = item.get('级别', '')
                        year = item.get('获得年份', '')
                        issuer = item.get('颁发机构', '')
                        
                        # 根据级别确定颜色
                        if '国家' in level or '一等' in level or '特等' in level:
                            color = '#E74C3C'  # 红色
                            icon = '🥇'
                        elif '省部' in level or '二等' in level or '高级' in level:
                            color = '#F39C12'  # 橙色
                            icon = '🥈'
                        elif '市级' in level or '三等' in level or '中级' in level:
                            color = '#3498DB'  # 蓝色
                            icon = '🥉'
                        else:
                            color = '#95A5A6'  # 灰色
                            icon = '🏅'
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;
                                   border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="font-size: 1.5em; margin-right: 10px;">{icon}</span>
                                <div style="flex: 1;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <strong style="color: #333; font-size: 1.1em;">{item_name}</strong>
                                        <span style="background: {color}; color: white; padding: 4px 8px; border-radius: 12px; 
                                                    font-size: 0.8em; font-weight: bold;">{level}</span>
                                    </div>
                                    <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                                        <span style="margin-right: 15px;">📅 {year}年</span>
                                        <span>🏛️ {issuer}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("暂无奖项、职称、资质记录")
    
    with tab3:
        # 培训记录
        st.markdown("**培训认证记录**")
        
        if cadre.training_records:
            # 按年份排序
            sorted_trainings = sorted(cadre.training_records, key=lambda x: x.get('年份', 0), reverse=True)
            
            for training in sorted_trainings:
                year = training.get('年份', '')
                content = training.get('培训内容', '')
                organization = training.get('培训机构', '')
                hours = training.get('学时', 0)
                
                # 确定培训类型
                if '领导' in content or '管理' in content:
                    training_type = '领导力'
                    color = '#E74C3C'
                    icon = '👑'
                elif '技术' in content or '专业' in content:
                    training_type = '专业技能'
                    color = '#3498DB'
                    icon = '🔧'
                elif '安全' in content or '质量' in content:
                    training_type = '安全质量'
                    color = '#E67E22'
                    icon = '🛡️'
                else:
                    training_type = '综合培训'
                    color = '#9B59B6'
                    icon = '📚'
                
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;
                           border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
                            <strong style="color: #333; font-size: 1.1em;">{content}</strong>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: {color}20; color: {color}; padding: 4px 8px; border-radius: 12px; 
                                        font-size: 0.8em; font-weight: bold;">{training_type}</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; color: #666; font-size: 0.9em;">
                        <div>
                            <span style="margin-right: 20px;">🏢 {organization}</span>
                            <span>🗺️ {year}年</span>
                        </div>
                        <div style="color: {color}; font-weight: bold;">{hours}学时</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂无培训记录")
    
    with tab4:
        # 绩效表现
        st.markdown("**绩效表现记录**")
        
        if cadre.performance_history:
            # 按年份排序
            sorted_performance = sorted(cadre.performance_history, key=lambda x: x.get('年份', 0), reverse=True)
            
            # 绩效趋势图
            years = [str(p.get('年份', '')) for p in sorted_performance]
            grades = [p.get('考核等级', '') for p in sorted_performance]
            
            # 转换成数字用于图表
            grade_scores = []
            for grade in grades:
                if grade == '优秀':
                    grade_scores.append(95)
                elif grade == '良好':
                    grade_scores.append(85)
                elif grade == '合格':
                    grade_scores.append(75)
                else:
                    grade_scores.append(65)
            
            # 绩效趋势图
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years[::-1],  # 反转以显示正序
                y=grade_scores[::-1],
                mode='lines+markers',
                name='绩效趋势',
                line=dict(color='#3498DB', width=3),
                marker=dict(size=8, color='#3498DB')
            ))
            
            fig.update_layout(
                title='绩效表现趋势',
                xaxis_title='年份',
                yaxis_title='绩效分数',
                yaxis=dict(range=[60, 100]),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 绩效详情
            st.markdown("**绩效详情**")
            
            for perf in sorted_performance:
                year = perf.get('年份', '')
                grade = perf.get('考核等级', '')
                achievements = perf.get('关键成果', '')
                suggestions = perf.get('改进建议', '')
                
                # 等级颜色
                if grade == '优秀':
                    grade_color = '#27AE60'
                    grade_icon = '🎆'
                elif grade == '良好':
                    grade_color = '#3498DB'
                    grade_icon = '👍'
                elif grade == '合格':
                    grade_color = '#F39C12'
                    grade_icon = '✅'
                else:
                    grade_color = '#95A5A6'
                    grade_icon = '📈'
                
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;
                           border-left: 4px solid {grade_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #333;">{year}年度考核</h4>
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.2em; margin-right: 5px;">{grade_icon}</span>
                            <span style="background: {grade_color}20; color: {grade_color}; padding: 6px 12px; 
                                        border-radius: 15px; font-weight: bold;">{grade}</span>
                        </div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #333;">关键成果：</strong>
                        <div style="color: #555; margin-top: 3px; line-height: 1.4;">{achievements}</div>
                    </div>
                    <div>
                        <strong style="color: #333;">改进建议：</strong>
                        <div style="color: #555; margin-top: 3px; line-height: 1.4;">{suggestions}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂无绩效记录")

def show_river_competency_model(data_manager: DataManager, river_model: RiverCompetencyModel):
    """显示江河胜任图模型 - 回退简洁版本并微调"""
    st.header("🌊 江河胜任图模型")
    st.markdown("*基于三峡物资招投标业务特色的能力评估模型*")
    
    cadres = data_manager.get_all_cadres()
    cadre_names = [cadre.name for cadre in cadres]
    selected_cadre_name = st.selectbox("选择干部", cadre_names)
    
    cadre = data_manager.get_cadre_by_name(selected_cadre_name)
    
    if cadre:
        # 简洁的信息展示
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("姓名", cadre.name)
        with col2:
            st.metric("部门", cadre.department)
        with col3:
            st.metric("职位", cadre.position)
        with col4:
            avg_score = sum(cadre.river_competency.values()) / len(cadre.river_competency) if cadre.river_competency else 0
            st.metric("综合评分", f"{avg_score:.1f}")
        
        st.subheader("📊 能力分析")
        
        # 图表区域
        col1, col2 = st.columns(2)
        
        with col1:
            competency_names = list(cadre.river_competency.keys())
            competency_scores = list(cadre.river_competency.values())
            
            # 江河胜任图
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=competency_scores,
                theta=competency_names,
                fill='toself',
                name=cadre.name,
                line=dict(color='#1f77b4', width=2),
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            
            fig.update_layout(
                title="🌊 江河胜任图",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dev_names = list(cadre.development_dimensions.keys())
            dev_scores = list(cadre.development_dimensions.values())
            
            # 发展维度图
            fig_dev = go.Figure()
            fig_dev.add_trace(go.Scatterpolar(
                r=dev_scores,
                theta=dev_names,
                fill='toself',
                name=cadre.name,
                line=dict(color='#ff7f0e', width=2),
                fillcolor='rgba(255, 127, 14, 0.3)'
            ))
            
            fig_dev.update_layout(
                title="📈 发展维度",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_dev, use_container_width=True)
        
        # 美化的能力分析卡片区域
        st.markdown("""
        <div style="margin: 30px 0;">
            <h3 style="text-align: center; color: #333; margin-bottom: 25px; font-weight: 600;">
                🎯 综合能力评估
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 核心评估指标
        eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
        
        with eval_col1:
            avg_competency = sum(competency_scores) / len(competency_scores) if competency_scores else 0
            if avg_competency >= 85:
                comp_level = "专家级别"
                comp_color = "#1abc9c"
                comp_icon = "🏆"
            elif avg_competency >= 75:
                comp_level = "高级水平"
                comp_color = "#3498db"
                comp_icon = "🔥"
            elif avg_competency >= 65:
                comp_level = "中级水平"
                comp_color = "#f39c12"
                comp_icon = "👍"
            else:
                comp_level = "初级水平"
                comp_color = "#e74c3c"
                comp_icon = "📈"
                
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {comp_color} 0%, {comp_color}CC 100%);
                       color: white; padding: 25px; border-radius: 15px; text-align: center;
                       box-shadow: 0 10px 25px {comp_color}40; margin-bottom: 15px;
                       transform: translateY(0); transition: all 0.3s ease;">
                <div style="font-size: 2.5em; margin-bottom: 10px;">{comp_icon}</div>
                <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 8px; opacity: 0.9;">{comp_level}</div>
                <div style="font-size: 2.8em; font-weight: 700; margin-bottom: 5px;">{avg_competency:.0f}</div>
                <div style="font-size: 0.9em; opacity: 0.8;">江河胜任图</div>
            </div>
            """, unsafe_allow_html=True)
        
        with eval_col2:
            avg_dev_score = sum(dev_scores) / len(dev_scores) if dev_scores else 0
            if avg_dev_score >= 85:
                dev_level = "发展优秀"
                dev_color = "#9b59b6"
                dev_icon = "🚀"
            elif avg_dev_score >= 75:
                dev_level = "发展良好"
                dev_color = "#34495e"
                dev_icon = "📈"
            elif avg_dev_score >= 65:
                dev_level = "发展中等"
                dev_color = "#f39c12"
                dev_icon = "⚡"
            else:
                dev_level = "需要发展"
                dev_color = "#e74c3c"
                dev_icon = "🎯"
                
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {dev_color} 0%, {dev_color}CC 100%);
                       color: white; padding: 25px; border-radius: 15px; text-align: center;
                       box-shadow: 0 10px 25px {dev_color}40; margin-bottom: 15px;">
                <div style="font-size: 2.5em; margin-bottom: 10px;">{dev_icon}</div>
                <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 8px; opacity: 0.9;">{dev_level}</div>
                <div style="font-size: 2.8em; font-weight: 700; margin-bottom: 5px;">{avg_dev_score:.0f}</div>
                <div style="font-size: 0.9em; opacity: 0.8;">发展维度</div>
            </div>
            """, unsafe_allow_html=True)
        
        with eval_col3:
            # 能力排名
            all_cadres = data_manager.get_all_cadres()
            if competency_scores:
                all_comp_scores = []
                for c in all_cadres:
                    if c.river_competency:
                        comp_avg = sum(c.river_competency.values()) / len(c.river_competency)
                        all_comp_scores.append((c.name, comp_avg))
                
                if all_comp_scores:
                    all_comp_scores.sort(key=lambda x: x[1], reverse=True)
                    current_rank = next((i+1 for i, (name, _) in enumerate(all_comp_scores) if name == cadre.name), 0)
                    
                    if current_rank <= len(all_comp_scores) * 0.2:
                        rank_color = "#e67e22"
                        rank_icon = "🥇"
                        rank_level = "顶尖水平"
                    elif current_rank <= len(all_comp_scores) * 0.5:
                        rank_color = "#27ae60"
                        rank_icon = "🥈"
                        rank_level = "优秀水平"
                    else:
                        rank_color = "#95a5a6"
                        rank_icon = "🥉"
                        rank_level = "一般水平"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {rank_color} 0%, {rank_color}CC 100%);
                               color: white; padding: 25px; border-radius: 15px; text-align: center;
                               box-shadow: 0 10px 25px {rank_color}40; margin-bottom: 15px;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">{rank_icon}</div>
                        <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 8px; opacity: 0.9;">{rank_level}</div>
                        <div style="font-size: 2.8em; font-weight: 700; margin-bottom: 5px;">{current_rank}</div>
                        <div style="font-size: 0.9em; opacity: 0.8;">排名/{len(all_comp_scores)}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with eval_col4:
            # 综合发展潜力
            potential_score = (avg_competency + avg_dev_score) / 2
            if potential_score >= 85:
                pot_level = "高潜力"
                pot_color = "#8e44ad"
                pot_icon = "🌟"
            elif potential_score >= 75:
                pot_level = "中高潜力"
                pot_color = "#2980b9"
                pot_icon = "⭐"
            elif potential_score >= 65:
                pot_level = "中等潜力"
                pot_color = "#f39c12"
                pot_icon = "✨"
            else:
                pot_level = "待发掘"
                pot_color = "#e74c3c"
                pot_icon = "💫"
                
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {pot_color} 0%, {pot_color}CC 100%);
                       color: white; padding: 25px; border-radius: 15px; text-align: center;
                       box-shadow: 0 10px 25px {pot_color}40; margin-bottom: 15px;">
                <div style="font-size: 2.5em; margin-bottom: 10px;">{pot_icon}</div>
                <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 8px; opacity: 0.9;">{pot_level}</div>
                <div style="font-size: 2.8em; font-weight: 700; margin-bottom: 5px;">{potential_score:.0f}</div>
                <div style="font-size: 0.9em; opacity: 0.8;">发展潜力</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 优雅的能力详情分析
        st.markdown("""
        <div style="margin: 40px 0 30px 0;">
            <h3 style="text-align: center; color: #333; margin-bottom: 30px; font-weight: 600;">
                📋 能力详情解析
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 能力强项与提升空间
        strength_col, improvement_col = st.columns(2)
        
        with strength_col:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
                       color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;
                       box-shadow: 0 8px 25px rgba(39, 174, 96, 0.3);">
                <h4 style="margin: 0 0 15px 0; text-align: center; font-weight: 600;">
                    💪 能力强项
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            if competency_scores:
                sorted_competencies = sorted(cadre.river_competency.items(), key=lambda x: x[1], reverse=True)
                for i, (comp, score) in enumerate(sorted_competencies[:3]):
                    st.markdown(f"""
                    <div style="background: white; padding: 18px; border-radius: 12px; margin-bottom: 12px;
                               border: 2px solid #27ae60; box-shadow: 0 4px 15px rgba(39, 174, 96, 0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <div style="font-weight: 600; font-size: 1.1em; color: #333; margin-bottom: 5px;">
                                    {i+1}. {comp}
                                </div>
                                <div style="width: 100%; background: #ecf0f1; border-radius: 10px; height: 8px;">
                                    <div style="width: {score}%; background: linear-gradient(90deg, #27ae60, #2ecc71);
                                               border-radius: 10px; height: 8px;"></div>
                                </div>
                            </div>
                            <div style="margin-left: 20px; text-align: center;">
                                <div style="font-size: 1.8em; font-weight: 700; color: #27ae60;">{score:.0f}</div>
                                <div style="font-size: 0.8em; color: #666;">分</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with improvement_col:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                       color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;
                       box-shadow: 0 8px 25px rgba(231, 76, 60, 0.3);">
                <h4 style="margin: 0 0 15px 0; text-align: center; font-weight: 600;">
                    📈 提升空间
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            if competency_scores:
                sorted_competencies = sorted(cadre.river_competency.items(), key=lambda x: x[1])
                for i, (comp, score) in enumerate(sorted_competencies[:3]):
                    st.markdown(f"""
                    <div style="background: white; padding: 18px; border-radius: 12px; margin-bottom: 12px;
                               border: 2px solid #e74c3c; box-shadow: 0 4px 15px rgba(231, 76, 60, 0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <div style="font-weight: 600; font-size: 1.1em; color: #333; margin-bottom: 5px;">
                                    {i+1}. {comp}
                                </div>
                                <div style="width: 100%; background: #ecf0f1; border-radius: 10px; height: 8px;">
                                    <div style="width: {score}%; background: linear-gradient(90deg, #e74c3c, #c0392b);
                                               border-radius: 10px; height: 8px;"></div>
                                </div>
                            </div>
                            <div style="margin-left: 20px; text-align: center;">
                                <div style="font-size: 1.8em; font-weight: 700; color: #e74c3c;">{score:.0f}</div>
                                <div style="font-size: 0.8em; color: #666;">分</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 详细的能力分析标签页
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🌊 江河胜任图详情", "📈 发展维度详情", "🎯 能力发展建议"])
        
        with tab1:
            st.markdown("""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                <h4 style="color: #667eea; margin: 0 0 15px 0; text-align: center;">江河胜任图各维度评分</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for competency, score in cadre.river_competency.items():
                level = river_model.get_competency_level(score)
                level_colors = {
                    "初级": "#e74c3c",
                    "中级": "#f39c12", 
                    "高级": "#3498db",
                    "专家": "#27ae60"
                }
                level_icons = {
                    "初级": "🌱",
                    "中级": "🌿", 
                    "高级": "🌳",
                    "专家": "🏆"
                }
                
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 15px; margin-bottom: 15px;
                           border: 2px solid {level_colors[level]}; 
                           box-shadow: 0 6px 20px {level_colors[level]}20;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="font-size: 1.5em; margin-right: 10px;">{level_icons[level]}</span>
                                <div>
                                    <div style="font-weight: 600; font-size: 1.2em; color: #333;">{competency}</div>
                                    <div style="font-size: 0.9em; color: {level_colors[level]}; font-weight: 600;">{level}水平</div>
                                </div>
                            </div>
                            <div style="width: 100%; background: #ecf0f1; border-radius: 10px; height: 10px;">
                                <div style="width: {score}%; background: linear-gradient(90deg, {level_colors[level]}, {level_colors[level]}88);
                                           border-radius: 10px; height: 10px; transition: width 0.5s ease;"></div>
                            </div>
                        </div>
                        <div style="margin-left: 25px; text-align: center;">
                            <div style="font-size: 2.5em; font-weight: 700; color: {level_colors[level]};">{score:.0f}</div>
                            <div style="font-size: 0.9em; color: #666;">/ 100</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div style="background: rgba(231, 76, 60, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                <h4 style="color: #e74c3c; margin: 0 0 15px 0; text-align: center;">个人发展维度评分</h4>
            </div>
            """, unsafe_allow_html=True)
            
            dev_colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6"]
            dev_icons = ["🎯", "⚡", "🚀", "💡", "🌟"]
            
            for i, (dimension, score) in enumerate(cadre.development_dimensions.items()):
                color = dev_colors[i % len(dev_colors)]
                icon = dev_icons[i % len(dev_icons)]
                
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 15px; margin-bottom: 15px;
                           border: 2px solid {color}; box-shadow: 0 6px 20px {color}20;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="font-size: 1.5em; margin-right: 10px;">{icon}</span>
                                <div style="font-weight: 600; font-size: 1.2em; color: #333;">{dimension}</div>
                            </div>
                            <div style="width: 100%; background: #ecf0f1; border-radius: 10px; height: 10px;">
                                <div style="width: {score}%; background: linear-gradient(90deg, {color}, {color}88);
                                           border-radius: 10px; height: 10px; transition: width 0.5s ease;"></div>
                            </div>
                        </div>
                        <div style="margin-left: 25px; text-align: center;">
                            <div style="font-size: 2.5em; font-weight: 700; color: {color};">{score:.0f}</div>
                            <div style="font-size: 0.9em; color: #666;">/ 100</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div style="background: rgba(155, 89, 182, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                <h4 style="color: #9b59b6; margin: 0 0 15px 0; text-align: center;">个性化发展建议</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 生成个性化建议
            if competency_scores:
                sorted_competencies = sorted(cadre.river_competency.items(), key=lambda x: x[1])
                weakest_areas = [comp for comp, score in sorted_competencies[:2]]
                
                suggestions = {
                    "招投标流程管理": "建议加强招投标法规学习，参与更多大型项目的流程管控工作",
                    "供应商关系维护": "建议建立系统的供应商管理制度，加强沟通协调技能培训",
                    "合同管理": "建议学习最新的合同法规，提升合同风险识别和防控能力",
                    "质量控制": "建议参加质量管理体系培训，建立完善的质量控制流程",
                    "成本控制": "建议学习成本分析方法，掌握先进的成本控制工具和技术",
                    "风险防控": "建议加强风险管理知识学习，建立系统的风险预警机制"
                }
                
                for i, area in enumerate(weakest_areas, 1):
                    suggestion = suggestions.get(area, "建议针对该能力开展专项培训和实践锻炼")
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; margin-bottom: 15px;
                               border-left: 5px solid #9b59b6; box-shadow: 0 4px 15px rgba(155, 89, 182, 0.1);">
                        <div style="display: flex; align-items: flex-start;">
                            <div style="background: #9b59b6; color: white; border-radius: 50%;
                                       width: 30px; height: 30px; display: flex; align-items: center;
                                       justify-content: center; margin-right: 15px; font-weight: bold;">{i}</div>
                            <div style="flex: 1;">
                                <div style="font-weight: 600; font-size: 1.1em; color: #333; margin-bottom: 8px;">
                                    {area} 提升建议
                                </div>
                                <div style="color: #555; line-height: 1.6;">
                                    {suggestion}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 详细能力分析
        st.subheader("📊 详细能力分析")
        
        for dimension, capabilities in river_model.competency_dimensions.items():
            with st.expander(f"{dimension} - {cadre.river_competency.get(dimension, 0):.1f}分"):
                st.markdown(f"**{dimension}包含以下能力要素：**")
                
                for capability, config in capabilities.items():
                    st.markdown(f"• **{capability}** (权重: {config['权重']*100:.0f}%)")
                    st.markdown(f"  - {config['描述']}")
                    
                    # 模拟单项能力分数
                    capability_score = cadre.river_competency.get(dimension, 70) + random.randint(-10, 10)
                    capability_score = max(0, min(100, capability_score))
                    
                    progress_color = "#4ECDC4" if capability_score >= 75 else "#FECA57" if capability_score >= 60 else "#FF6B6B"
                    st.markdown(f"""
                    <div style="background-color: {progress_color}; height: 20px; border-radius: 10px; width: {capability_score}%; margin: 5px 0;"></div>
                    <p style="margin: 0; font-size: 0.8em;">{capability_score:.1f}分</p>
                    """, unsafe_allow_html=True)

def show_position_matching(data_manager: DataManager, matching_engine: PositionMatchingEngine):
    """显示智能人岗匹配"""
    st.header("🎯 智能人岗匹配")
    st.markdown("*基于江河胜任图的岗位适配分析*")
    
    cadres = data_manager.get_all_cadres()
    cadre_names = [cadre.name for cadre in cadres]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_cadre_name = st.selectbox("选择干部", cadre_names)
    
    with col2:
        positions = list(matching_engine.position_requirements.keys())
        selected_position = st.selectbox("选择目标岗位", positions)
    
    if st.button("🔍 开始匹配分析", type="primary"):
        cadre = data_manager.get_cadre_by_name(selected_cadre_name)
        
        if cadre:
            # 计算匹配度
            match_result = matching_engine.calculate_position_match(cadre, selected_position)
            
            # 显示匹配结果
            col1, col2, col3 = st.columns(3)
            
            with col1:
                match_score = match_result["匹配度"]
                color = "#4ECDC4" if match_score >= 75 else "#FECA57" if match_score >= 60 else "#FF6B6B"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 匹配度</h3>
                    <h1 style="color: {color};">{match_score:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                assessment = match_result["综合评价"]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 综合评价</h3>
                    <h2 style="color: {color};">{assessment}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                recommendation_count = len(match_result["改进建议"])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💡 改进建议</h3>
                    <h1 style="color: #667eea;">{recommendation_count}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # 详细匹配分析
            st.subheader("📈 详细匹配分析")
            
            match_details = match_result["匹配详情"]
            
            # 创建匹配详情图表
            categories = list(match_details.keys())
            scores = list(match_details.values())
            
            fig = px.bar(
                x=categories,
                y=scores,
                title="各维度匹配分析",
                color=scores,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_title="匹配维度", yaxis_title="得分")
            st.plotly_chart(fig, use_container_width=True)
            
            # 改进建议
            if match_result["改进建议"]:
                st.subheader("💡 改进建议")
                for i, suggestion in enumerate(match_result["改进建议"], 1):
                    st.markdown(f"""
                    <div class="warning-card">
                        <strong>{i}. {suggestion}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-card">
                    <strong>🎉 恭喜！该干部完全符合岗位要求，无需改进。</strong>
                </div>
                """, unsafe_allow_html=True)
    
    # 岗位要求详情
    st.subheader("📋 岗位要求详情")
    
    with st.expander(f"查看{selected_position}详细要求"):
        requirements = matching_engine.position_requirements[selected_position]
        
        st.markdown("**核心能力要求：**")
        for competency in requirements["核心能力"]:
            st.markdown(f"• {competency}")
        
        st.markdown("**最低要求：**")
        for req, value in requirements["最低要求"].items():
            st.markdown(f"• {req}: {value}")
        
        st.markdown("**优先条件：**")
        for condition, values in requirements["优先条件"].items():
            st.markdown(f"• {condition}: {', '.join(values)}")

def show_data_visualization(data_manager: DataManager):
    """显示数据可视化分析"""
    st.header("📈 数据可视化分析")
    
    cadres = data_manager.get_all_cadres()
    
    # 创建DataFrame
    data = []
    for cadre in cadres:
        data.append({
            "姓名": cadre.name,
            "年龄": cadre.age,
            "工作年限": cadre.work_years,
            "素质基础": cadre.quality_foundation,
            "胜任能力": cadre.competency,
            "工作绩效": cadre.performance,
            "自画像": cadre.self_portrait,
            "声誉得分": cadre.reputation_score,
            "部门": cadre.department,
            "教育背景": cadre.education,
            "职位": cadre.position
        })
    
    df = pd.DataFrame(data)
    
    # 选择可视化类型
    viz_type = st.selectbox("选择可视化类型", [
        "📊 五维画像对比",
        "📈 年龄与能力关系",
        "🎯 部门能力分析",
        "📋 综合数据表格"
    ])
    
    if viz_type == "📊 五维画像对比":
        st.subheader("五维画像对比分析")
        
        # 选择要对比的干部
        selected_cadres = st.multiselect("选择对比的干部", df["姓名"].tolist(), default=df["姓名"].tolist()[:3])
        
        if selected_cadres:
            # 创建对比图表
            dimensions = ["素质基础", "胜任能力", "工作绩效", "自画像", "声誉得分"]
            
            fig = go.Figure()
            
            for cadre_name in selected_cadres:
                cadre_data = df[df["姓名"] == cadre_name].iloc[0]
                scores = [cadre_data[dim] for dim in dimensions]
                
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=dimensions,
                    fill='toself',
                    name=cadre_name
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="五维画像对比分析"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "📈 年龄与能力关系":
        st.subheader("年龄与能力关系分析")
        
        # 选择分析维度
        selected_dimension = st.selectbox("选择分析维度", ["素质基础", "胜任能力", "工作绩效", "自画像", "声誉得分"])
        
        fig = px.scatter(
            df, 
            x="年龄", 
            y=selected_dimension,
            size="工作年限",
            color="部门",
            hover_data=["姓名", "职位"],
            title=f"年龄与{selected_dimension}关系分析"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 相关性分析
        correlation = df["年龄"].corr(df[selected_dimension])
        st.markdown(f"**相关性系数：** {correlation:.3f}")
        
        if abs(correlation) > 0.5:
            st.success("存在较强相关性")
        elif abs(correlation) > 0.3:
            st.warning("存在中等相关性")
        else:
            st.info("相关性较弱")
    
    elif viz_type == "🎯 部门能力分析":
        st.subheader("部门能力分析")
        
        # 按部门分组分析
        dept_analysis = df.groupby("部门")[["素质基础", "胜任能力", "工作绩效", "自画像", "声誉得分"]].mean()
        
        fig = go.Figure()
        
        dimensions = ["素质基础", "胜任能力", "工作绩效", "自画像", "声誉得分"]
        
        for dept in dept_analysis.index:
            scores = [dept_analysis.loc[dept, dim] for dim in dimensions]
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions,
                fill='toself',
                name=dept
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="部门平均能力分析"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 部门排名
        st.subheader("📊 部门综合排名")
        
        dept_analysis["综合得分"] = dept_analysis.mean(axis=1)
        dept_ranking = dept_analysis.sort_values("综合得分", ascending=False)
        
        for i, (dept, score) in enumerate(dept_ranking["综合得分"].items(), 1):
            st.markdown(f"**{i}. {dept}** - {score:.1f}分")
    
    elif viz_type == "📋 综合数据表格":
        st.subheader("综合数据表格")
        
        # 可编辑表格
        st.dataframe(df, use_container_width=True)
        
        # 统计摘要
        st.subheader("📊 统计摘要")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**数值型变量统计：**")
            numeric_columns = ["年龄", "工作年限", "素质基础", "胜任能力", "工作绩效", "自画像", "声誉得分"]
            st.dataframe(df[numeric_columns].describe())
        
        with col2:
            st.markdown("**分类型变量统计：**")
            categorical_columns = ["部门", "教育背景", "职位"]
            for col in categorical_columns:
                st.markdown(f"**{col}分布：**")
                st.write(df[col].value_counts())

def show_development_suggestions(data_manager: DataManager, river_model: RiverCompetencyModel):
    """显示人才发展建议"""
    st.header("🔍 人才发展建议")
    st.markdown("*基于江河胜任图的个性化发展路径*")
    
    cadres = data_manager.get_all_cadres()
    cadre_names = [cadre.name for cadre in cadres]
    
    selected_cadre_name = st.selectbox("选择干部", cadre_names)
    cadre = data_manager.get_cadre_by_name(selected_cadre_name)
    
    if cadre:
        # 能力诊断
        st.subheader("🔍 能力诊断")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**优势能力：**")
            sorted_competencies = sorted(cadre.river_competency.items(), key=lambda x: x[1], reverse=True)
            
            for competency, score in sorted_competencies[:2]:
                if score >= 75:
                    st.markdown(f"""
                    <div class="success-card">
                        <strong>{competency}</strong><br>
                        得分: {score:.1f} ({river_model.get_competency_level(score)})
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**提升空间：**")
            
            for competency, score in sorted_competencies[-2:]:
                if score < 75:
                    st.markdown(f"""
                    <div class="warning-card">
                        <strong>{competency}</strong><br>
                        得分: {score:.1f} ({river_model.get_competency_level(score)})
                    </div>
                    """, unsafe_allow_html=True)
        
        # 发展路径建议
        st.subheader("🛤️ 发展路径建议")
        
        # 基于当前能力水平给出建议
        avg_competency = sum(cadre.river_competency.values()) / len(cadre.river_competency)
        
        if avg_competency >= 85:
            development_stage = "专家级发展"
            suggestions = [
                "担任技术专家或顾问角色",
                "主导重要项目和创新实践",
                "培养和指导团队成员",
                "参与行业标准制定",
                "拓展跨领域合作"
            ]
        elif avg_competency >= 75:
            development_stage = "高级发展"
            suggestions = [
                "承担更多管理责任",
                "参与复杂项目的决策",
                "提升跨部门协调能力",
                "加强专业深度发展",
                "建立行业影响力"
            ]
        elif avg_competency >= 65:
            development_stage = "中级发展"
            suggestions = [
                "加强专业技能培训",
                "参与中等复杂度项目",
                "提升团队协作能力",
                "学习新技术和方法",
                "建立专业网络"
            ]
        else:
            development_stage = "基础发展"
            suggestions = [
                "完善基础专业知识",
                "参与基础项目实践",
                "加强基本技能训练",
                "寻求导师指导",
                "建立学习习惯"
            ]
        
        st.markdown(f"**发展阶段：** {development_stage}")
        
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"{i}. {suggestion}")
        
        # 培训建议
        st.subheader("📚 培训建议")
        
        training_recommendations = []
        
        # 基于薄弱环节推荐培训
        weakest_competency = min(cadre.river_competency.items(), key=lambda x: x[1])
        
        training_map = {
            "招投标管理能力": [
                "招投标法律法规培训",
                "招投标实务操作培训",
                "供应商管理培训",
                "合同管理培训"
            ],
            "物资管理能力": [
                "供应链管理培训",
                "库存管理培训",
                "采购管理培训",
                "物流管理培训"
            ],
            "风险管理能力": [
                "风险识别与评估培训",
                "内控制度建设培训",
                "合规管理培训",
                "应急管理培训"
            ],
            "技术创新能力": [
                "数字化转型培训",
                "项目管理培训",
                "创新方法培训",
                "技术发展趋势培训"
            ]
        }
        
        if weakest_competency[0] in training_map:
            training_recommendations.extend(training_map[weakest_competency[0]])
        
        # 通用培训建议
        training_recommendations.extend([
            "领导力提升培训",
            "沟通技巧培训",
            "团队建设培训"
        ])
        
        for training in training_recommendations:
            st.markdown(f"• {training}")
        
        # 职业发展时间线
        st.subheader("📅 职业发展时间线")
        
        timeline_data = []
        current_year = datetime.now().year
        
        # 基于年龄和能力水平设计发展时间线
        if cadre.age < 35:
            timeline_data = [
                {"年份": current_year, "目标": "能力提升期", "描述": "专注专业能力发展"},
                {"年份": current_year + 2, "目标": "责任扩展期", "描述": "承担更多工作责任"},
                {"年份": current_year + 5, "目标": "管理转型期", "描述": "向管理岗位发展"},
                {"年份": current_year + 8, "目标": "专业专家期", "描述": "成为领域专家"}
            ]
        elif cadre.age < 45:
            timeline_data = [
                {"年份": current_year, "目标": "能力巩固期", "描述": "巩固现有能力优势"},
                {"年份": current_year + 2, "目标": "影响力扩展", "描述": "扩大专业影响力"},
                {"年份": current_year + 5, "目标": "资深专家期", "描述": "成为行业资深专家"}
            ]
        else:
            timeline_data = [
                {"年份": current_year, "目标": "经验传承期", "描述": "传承经验培养新人"},
                {"年份": current_year + 2, "目标": "顾问专家期", "描述": "担任顾问专家角色"}
            ]
        
        for item in timeline_data:
            st.markdown(f"**{item['年份']}年 - {item['目标']}**")
            st.markdown(f"  {item['描述']}")

def show_add_cadre_form(data_manager: DataManager, analyzer: PersonalMaterialAnalyzer, river_model: RiverCompetencyModel):
    """显示添加干部表单"""
    st.header("➕ 添加干部信息")
    
    with st.form("add_cadre_form"):
        st.subheader("基本信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("姓名*", placeholder="请输入姓名")
            age = st.number_input("年龄*", min_value=20, max_value=65, value=30)
            gender = st.selectbox("性别*", ["男", "女"])
            education = st.selectbox("教育背景*", ["专科", "本科", "硕士", "博士"])
        
        with col2:
            department = st.selectbox("部门*", [
                "招投标管理部", "物资管理部", "采购管理部", "技术创新部", 
                "风险管理部", "大水电事业部", "长江大保护部", 
                "抽水蓄能部", "新能源事业部", "电子商城部", 
                "供应链管理部", "综合管理部", "其他"
            ])
            position = st.text_input("职位*", placeholder="请输入职位")
            work_years = st.number_input("工作年限*", min_value=0, max_value=50, value=5)
            political_status = st.selectbox("政治面貌*", [
                "中共党员", "中共预备党员", "共青团员", "民主党派", "群众"
            ])
        
        st.subheader("个人详细材料")
        description = st.text_area(
            "个人详细描述*", 
            height=300,
            placeholder="请输入个人详细材料，包括政治素质、道德品行、工作能力、工作绩效、个人作风等方面的具体表现..."
        )
        
        submitted = st.form_submit_button("✅ 添加干部", type="primary")
        
        if submitted:
            if name and description:
                # 创建干部对象
                new_cadre = CadreProfile(
                    name=name,
                    age=age,
                    gender=gender,
                    education=education,
                    department=department,
                    position=position,
                    work_years=work_years,
                    political_status=political_status,
                    description=description
                )
                
                # 分析个人材料
                analyzer.analyze_personal_material(new_cadre)
                
                # 计算江河胜任图
                new_cadre.river_competency = river_model.calculate_river_competency(new_cadre)
                
                # 添加到数据管理器
                data_manager.add_cadre(new_cadre)
                
                st.success(f"✅ 成功添加干部 {name}！")
                
                # 显示分析结果
                st.subheader("📊 智能分析结果")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**干部画像分析：**")
                    dimensions = ["素质基础", "胜任能力", "工作绩效", "自画像", "声誉得分"]
                    scores = [new_cadre.quality_foundation, new_cadre.competency, 
                             new_cadre.performance, new_cadre.self_portrait, 
                             new_cadre.reputation_score]
                    
                    for dim, score in zip(dimensions, scores):
                        st.markdown(f"• {dim}: {score:.1f}")
                
                with col2:
                    st.markdown("**江河胜任图分析：**")
                    for competency, score in new_cadre.river_competency.items():
                        level = river_model.get_competency_level(score)
                        st.markdown(f"• {competency}: {score:.1f} ({level})")
                
                st.markdown("**提取的关键标签：**")
                if new_cadre.extracted_tags:
                    for dimension, categories in new_cadre.extracted_tags.items():
                        if any(tags for tags in categories.values()):
                            st.markdown(f"*{dimension}*: ", end="")
                            all_tags = []
                            for category, tags in categories.items():
                                all_tags.extend(tags)
                            if all_tags:
                                st.markdown(", ".join(all_tags))
                            else:
                                st.markdown("无相关标签")
                
            else:
                st.error("请填写所有必填项目（标有*的字段）")

def show_personal_work_network(data_manager: DataManager, cadre_names: list):
    """显示个人工作关系网络"""
    st.subheader("🏢 个人工作关系网络")
    st.markdown("*展示干部负责的项目和供应商信息*")
    
    selected_cadre_name = st.selectbox("选择干部", cadre_names)
    selected_cadre = data_manager.get_cadre_by_name(selected_cadre_name)
    
    if not selected_cadre:
        st.error("未找到选择的干部")
        return
        
    # 控制面板
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        if st.button("🔍 生成工作关系网络", type="primary", use_container_width=True):
            st.session_state["work_network_generated"] = True
            st.session_state["selected_work_cadre"] = selected_cadre_name
    
    with col_right:
        if st.button("🗑️ 清除网络显示", use_container_width=True):
            if "work_network_generated" in st.session_state:
                del st.session_state["work_network_generated"]
            if "selected_work_cadre" in st.session_state:
                del st.session_state["selected_work_cadre"]
    
    # 生成并显示工作关系网络
    if st.session_state.get("work_network_generated", False) and st.session_state.get("selected_work_cadre") == selected_cadre_name:
        # 生成项目和供应商数据
        import random
        projects_data = generate_work_projects_for_cadre(selected_cadre)
        suppliers_data = generate_suppliers_for_cadre(selected_cadre)
        
        # 统计信息
        st.markdown("### 📈 工作关系统计")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       color: white; padding: 20px; border-radius: 15px; text-align: center;
                       box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
                <div style="font-size: 1.2em; margin-bottom: 5px;">📋</div>
                <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{len(projects_data)}</div>
                <div style="font-size: 0.9em; opacity: 0.9;">负责项目</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                       color: white; padding: 20px; border-radius: 15px; text-align: center;
                       box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);">
                <div style="font-size: 1.2em; margin-bottom: 5px;">🏢</div>
                <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{len(suppliers_data)}</div>
                <div style="font-size: 0.9em; opacity: 0.9;">合作供应商</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_budget = sum(float(p['budget'].replace('万元', '').replace(',', '')) for p in projects_data)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                       color: white; padding: 20px; border-radius: 15px; text-align: center;
                       box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);">
                <div style="font-size: 1.2em; margin-bottom: 5px;">💰</div>
                <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{total_budget:.0f}</div>
                <div style="font-size: 0.9em; opacity: 0.9;">总预算(万元)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            high_value_projects = sum(1 for p in projects_data if float(p['budget'].replace('万元', '').replace(',', '')) > 1000)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                       color: white; padding: 20px; border-radius: 15px; text-align: center;
                       box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4);">
                <div style="font-size: 1.2em; margin-bottom: 5px;">⭐</div>
                <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{high_value_projects}</div>
                <div style="font-size: 0.9em; opacity: 0.9;">重点项目</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 智能过滤器
        st.markdown("### 🔧 智能过滤与分析")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            show_projects = st.checkbox("显示项目信息", value=True)
            project_budget_filter = st.selectbox("项目预算筛选", ["全部", "500万以上", "1000万以上", "5000万以上"])
        
        with filter_col2:
            show_suppliers = st.checkbox("显示供应商信息", value=True)
            supplier_rating_filter = st.selectbox("供应商评级筛选", ["全部", "A级", "B级", "C级"])
        
        with filter_col3:
            layout_style = st.selectbox("图谱布局", ["层次布局", "圆形布局", "力导向布局"])
            max_nodes = st.slider("最大显示节点数", 10, 50, 25)
        
        # 应用筛选
        filtered_projects = filter_projects(projects_data, project_budget_filter)
        filtered_suppliers = filter_suppliers(suppliers_data, supplier_rating_filter)
        
        # 优化后的网络图谱
        display_optimized_work_network(selected_cadre_name, filtered_projects, filtered_suppliers, 
                                     show_projects, show_suppliers, layout_style, max_nodes)
        
        # 详细信息展示
        display_work_details(filtered_projects, filtered_suppliers, show_projects, show_suppliers)

def show_personal_family_network(data_manager: DataManager, cadre_names: list):
    """显示个人亲属关系网络"""
    st.subheader("👨‍👩‍👧‍👦 个人亲属关系网络")
    st.markdown("*展示干部的父母、子女、配偶等家属信息*")
    
    selected_cadre_name = st.selectbox("选择干部", cadre_names, key="family_cadre_select")
    selected_cadre = data_manager.get_cadre_by_name(selected_cadre_name)
    
    if not selected_cadre:
        st.error("未找到选择的干部")
        return
    
    # 检查是否有家属信息
    if not hasattr(selected_cadre, 'family_relations') or not selected_cadre.family_relations:
        st.warning("该干部暂无家属关系信息")
        return
    
    family_relations = selected_cadre.family_relations
    
    # 家属关系统计
    st.markdown("### 👨‍👩‍👧‍👦 家属关系统计")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 15px; text-align: center;
                   box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
            <div style="font-size: 1.2em; margin-bottom: 5px;">👫</div>
            <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">2</div>
            <div style="font-size: 0.9em; opacity: 0.9;">父母</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        spouse_count = 1 if family_relations.get('spouse', {}).get('name') else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                   color: white; padding: 20px; border-radius: 15px; text-align: center;
                   box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);">
            <div style="font-size: 1.2em; margin-bottom: 5px;">💑</div>
            <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{spouse_count}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">配偶</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        children_count = len(family_relations.get('children', []))
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                   color: white; padding: 20px; border-radius: 15px; text-align: center;
                   box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);">
            <div style="font-size: 1.2em; margin-bottom: 5px;">👶</div>
            <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{children_count}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">子女</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_family = 2 + spouse_count + children_count
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                   color: white; padding: 20px; border-radius: 15px; text-align: center;
                   box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4);">
            <div style="font-size: 1.2em; margin-bottom: 5px;">👨‍👩‍👧‍👦</div>
            <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{total_family}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">家庭成员</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 显示家属关系网络图
    display_family_network_graph(selected_cadre_name, family_relations)
    
    # 详细家属信息
    display_family_details(family_relations)

def show_integrity_risk_assessment(data_manager: DataManager, cadre_names: list):
    """显示廉洁风险评估"""
    st.subheader("⚠️ 廉洁风险评估")
    st.markdown("*基于工作关系和家属关系的廉洁风险评估*")
    
    selected_cadre_name = st.selectbox("选择干部", cadre_names, key="risk_cadre_select")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 风险评估维度")
        risk_dimensions = st.multiselect(
            "选择评估维度",
            ["工作关系风险", "家庭关系风险", "职位权力风险", "历史行为风险", "外部关联风险"],
            default=["工作关系风险", "家庭关系风险", "职位权力风险"]
        )
    
    with col2:
        if st.button("🔍 开始风险评估", type="primary", use_container_width=True):
            st.session_state["risk_assessment_done"] = True
            st.session_state["risk_cadre"] = selected_cadre_name
            st.session_state["risk_dimensions"] = risk_dimensions
    
    # 显示风险评估结果
    if st.session_state.get("risk_assessment_done", False) and st.session_state.get("risk_cadre") == selected_cadre_name:
        risk_result = perform_integrity_risk_assessment(data_manager, selected_cadre_name, risk_dimensions)
        display_risk_assessment_results(risk_result)

def show_overall_relationship_analysis(data_manager: DataManager, cadre_names: list):
    """显示整体关系分析"""
    st.subheader("📊 整体关系分析")
    st.markdown("*全局干部关系网络分析与风险监控*")
    
    # 整体统计信息
    display_overall_statistics(data_manager, cadre_names)
    
    # 关系网络全景图
    display_overall_network_graph(data_manager, cadre_names)
    
    # 风险排行榜
    display_risk_ranking(data_manager, cadre_names)

# 辅助函数
def generate_work_projects_for_cadre(cadre):
    """为干部生成工作项目数据"""
    import random
    
    project_templates = [
        {"name": "白鹤滩水电站机电设备采购", "type": "水电工程", "budget_range": [8000, 15000]},
        {"name": "乌东德升船机设备招标", "type": "船舶工程", "budget_range": [3000, 8000]},
        {"name": "三峡枢纽维护设备采购", "type": "维护工程", "budget_range": [1000, 3000]},
        {"name": "溪洛渡电站改造项目", "type": "改造工程", "budget_range": [5000, 12000]},
        {"name": "向家坝导航设施建设", "type": "导航工程", "budget_range": [2000, 5000]},
        {"name": "长江清洁能源装备采购", "type": "清洁能源", "budget_range": [4000, 10000]},
        {"name": "数字化监控系统建设", "type": "信息化", "budget_range": [800, 2500]},
        {"name": "安全环保设备更新", "type": "安全环保", "budget_range": [1500, 4000]},
    ]
    
    # 根据干部职位生成相关项目
    num_projects = random.randint(3, 8)
    projects = []
    
    for i in range(num_projects):
        template = random.choice(project_templates)
        budget = random.randint(template["budget_range"][0], template["budget_range"][1])
        
        project = {
            "name": template["name"],
            "type": template["type"],
            "budget": f"{budget:,}万元",
            "tender_number": f"TGWZ-{random.randint(2020, 2024)}-{random.randint(100, 999)}",
            "role": random.choice(["项目负责人", "技术负责人", "采购负责人", "评标专家", "监督员"]),
            "participation_time": f"{random.randint(2020, 2024)}年{random.randint(1, 12)}月",
            "status": random.choice(["进行中", "已完成", "待开标", "评标中"])
        }
        projects.append(project)
    
    return projects

def generate_suppliers_for_cadre(cadre):
    """为干部生成供应商数据"""
    import random
    
    supplier_templates = [
        {"name": "哈尔滨电机厂有限责任公司", "type": "发电设备", "rating": "A"},
        {"name": "东方电机有限公司", "type": "发电设备", "rating": "A"},
        {"name": "上海电气集团股份有限公司", "type": "电力设备", "rating": "A"},
        {"name": "特变电工股份有限公司", "type": "变压器", "rating": "B"},
        {"name": "中国能建集团", "type": "工程建设", "rating": "A"},
        {"name": "中国电建集团", "type": "工程建设", "rating": "A"},
        {"name": "华为技术有限公司", "type": "信息化", "rating": "A"},
        {"name": "施耐德电气(中国)有限公司", "type": "自动化", "rating": "B"},
        {"name": "ABB(中国)有限公司", "type": "电力自动化", "rating": "A"},
        {"name": "西门子(中国)有限公司", "type": "工业自动化", "rating": "A"},
    ]
    
    num_suppliers = random.randint(4, 10)
    suppliers = []
    
    for i in range(num_suppliers):
        template = random.choice(supplier_templates)
        
        supplier = {
            "name": template["name"],
            "type": template["type"],
            "rating": template["rating"],
            "cooperation_content": random.choice(["设备供应", "技术服务", "工程承包", "维护保养", "技术咨询"]),
            "project_relation": random.choice(["白鹤滩项目", "乌东德项目", "三峡工程", "溪洛渡项目", "向家坝项目"]),
            "contact_frequency": random.choice(["频繁", "定期", "偶尔", "项目期间"]),
            "contract_amount": f"{random.randint(500, 5000)}万元"
        }
        suppliers.append(supplier)
    
    return suppliers

def filter_projects(projects_data, budget_filter):
    """根据预算筛选项目"""
    if budget_filter == "全部":
        return projects_data
    
    budget_map = {
        "500万以上": 500,
        "1000万以上": 1000,
        "5000万以上": 5000
    }
    
    min_budget = budget_map.get(budget_filter, 0)
    return [p for p in projects_data 
            if float(p['budget'].replace('万元', '').replace(',', '')) >= min_budget]

def filter_suppliers(suppliers_data, rating_filter):
    """根据评级筛选供应商"""
    if rating_filter == "全部":
        return suppliers_data
    
    return [s for s in suppliers_data if s['rating'] == rating_filter]

def display_optimized_work_network(cadre_name, projects, suppliers, show_projects, show_suppliers, layout_style, max_nodes):
    """显示优化的工作关系网络图 - 解决节点显示和布局问题"""
    import plotly.graph_objects as go
    import networkx as nx
    import random
    
    # 如果没有数据，显示提示信息
    if not show_projects and not show_suppliers:
        st.info("请选择显示项目信息或供应商信息来生成关系网络图")
        return
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加中心节点（干部）- 使用深色背景和白色文字
    G.add_node(cadre_name, type='cadre', label=cadre_name, color='#1a365d', size=50, text_color='white')
    
    # 添加项目节点（限制数量）
    project_nodes = []
    if show_projects:
        for i, project in enumerate(projects[:max_nodes//3]):
            node_id = f"project_{i}"
            # 优化标签显示
            label = project['name']
            if len(label) > 12:
                label = label[:10] + "..."
            G.add_node(node_id, type='project', label=label, 
                      color='#c41e3a', size=35, details=project, text_color='white')
            G.add_edge(cadre_name, node_id, relation="负责")
            project_nodes.append(node_id)
    
    # 添加供应商节点（限制数量）
    supplier_nodes = []
    if show_suppliers:
        for i, supplier in enumerate(suppliers[:max_nodes//3]):
            node_id = f"supplier_{i}"
            # 优化标签显示
            label = supplier['name']
            if len(label) > 12:
                label = label[:10] + "..."
            G.add_node(node_id, type='supplier', label=label, 
                      color='#2e7d32', size=30, details=supplier, text_color='white')
            G.add_edge(cadre_name, node_id, relation="合作")
            supplier_nodes.append(node_id)
    
    # 改进项目与供应商之间的关系 - 减少连接数量避免混乱
    if show_projects and show_suppliers and project_nodes and supplier_nodes:
        # 为每个项目关联更少的供应商，减少视觉混乱
        for project_id in project_nodes:
            num_suppliers = min(random.randint(1, 2), len(supplier_nodes))
            connected_suppliers = random.sample(supplier_nodes, num_suppliers)
            for supplier_id in connected_suppliers:
                G.add_edge(project_id, supplier_id, relation="供应")
    
    # 优化布局算法 - 减少节点重叠和混乱
    try:
        if layout_style == "层次布局":
            # 使用更大的k值和更多迭代次数
            pos = nx.spring_layout(G, k=3.5, iterations=200, seed=42)
        elif layout_style == "圆形布局":
            pos = nx.circular_layout(G, scale=2.5)
        else:  # 力导向布局
            pos = nx.spring_layout(G, k=3.0, iterations=150, seed=42)
    except:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # 创建边的traces - 优化连线显示
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # 根据连接类型设置不同的线条样式
        relation = G.edges[edge].get('relation', '')
        if relation == '供应':
            line_color = 'rgba(100, 100, 100, 0.3)'
            line_width = 1
        else:
            line_color = 'rgba(150, 150, 150, 0.6)'
            line_width = 2
            
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=line_width, color=line_color),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # 分类节点
    cadre_nodes = []
    project_nodes = []
    supplier_nodes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_info = G.nodes[node]
        
        if node_info['type'] == 'cadre':
            cadre_nodes.append((x, y, node_info))
        elif node_info['type'] == 'project':
            project_nodes.append((x, y, node_info))
        elif node_info['type'] == 'supplier':
            supplier_nodes.append((x, y, node_info))
    
    traces = edge_traces.copy()
    
    # 干部节点 - 使用深色背景和白色文字
    if cadre_nodes:
        x_vals, y_vals, infos = zip(*cadre_nodes)
        traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            text=[info['label'] for info in infos],
            textposition="middle center",
            textfont=dict(size=18, color='#000000', family='Arial Black, sans-serif', weight='bold'),
            marker=dict(
                size=[info['size'] for info in infos],
                color=[info['color'] for info in infos],
                line=dict(width=4, color='#000000'),
                opacity=1.0
            ),
            hoverinfo='text',
            hovertext=[f"<b>{info['label']}</b><br>身份: 干部<br>类型: 中心节点" for info in infos],
            name="干部",
            showlegend=False
        ))
    
    # 项目节点 - 使用深红色背景和白色文字
    if project_nodes:
        x_vals, y_vals, infos = zip(*project_nodes)
        hover_texts = []
        for info in infos:
            details = info.get('details', {})
            hover_text = f"<b>{details.get('name', info['label'])}</b><br>"
            hover_text += f"类型: 项目<br>"
            hover_text += f"预算: {details.get('budget', '未知')}<br>"
            hover_text += f"状态: {details.get('status', '未知')}"
            hover_texts.append(hover_text)
        
        traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            text=[info['label'] for info in infos],
            textposition="middle center",
            textfont=dict(size=14, color='#000000', family='Arial Black, sans-serif', weight='bold'),
            marker=dict(
                size=[info['size'] for info in infos],
                color=[info['color'] for info in infos],
                line=dict(width=3, color='#000000'),
                opacity=1.0
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            name="项目",
            showlegend=False
        ))
    
    # 供应商节点 - 使用深绿色背景和白色文字
    if supplier_nodes:
        x_vals, y_vals, infos = zip(*supplier_nodes)
        hover_texts = []
        for info in infos:
            details = info.get('details', {})
            hover_text = f"<b>{details.get('name', info['label'])}</b><br>"
            hover_text += f"类型: 供应商<br>"
            hover_text += f"评级: {details.get('rating', '未知')}<br>"
            hover_text += f"合作内容: {details.get('cooperation_content', '未知')}"
            hover_texts.append(hover_text)
        
        traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            text=[info['label'] for info in infos],
            textposition="middle center",
            textfont=dict(size=13, color='#000000', family='Arial Black, sans-serif', weight='bold'),
            marker=dict(
                size=[info['size'] for info in infos],
                color=[info['color'] for info in infos],
                line=dict(width=3, color='#000000'),
                opacity=1.0
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            name="供应商",
            showlegend=False
        ))
    
    # 创建图表
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=dict(
            text=f'🌐 {cadre_name} 的工作关系网络',
            font=dict(size=20, color='#2e2e2e', family='Arial Black')
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=80, l=60, r=60, t=80),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f8f9fa',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        font=dict(family='Arial, sans-serif')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_work_details(projects, suppliers, show_projects, show_suppliers):
    """显示工作关系详细信息"""
    st.markdown("### 📋 详细信息")
    
    tab1, tab2 = st.tabs(["📋 项目详情", "🏢 供应商详情"])
    
    with tab1:
        if show_projects and projects:
            for project in projects:
                with st.expander(f"📋 {project['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**招标编号:** {project['tender_number']}")
                        st.markdown(f"**项目类型:** {project['type']}")
                        st.markdown(f"**参与角色:** {project['role']}")
                    with col2:
                        st.markdown(f"**项目预算:** {project['budget']}")
                        st.markdown(f"**参与时间:** {project['participation_time']}")
                        st.markdown(f"**项目状态:** {project['status']}")
        else:
            st.info("未选择显示项目信息或无项目数据")
    
    with tab2:
        if show_suppliers and suppliers:
            for supplier in suppliers:
                with st.expander(f"🏢 {supplier['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**供应商类型:** {supplier['type']}")
                        st.markdown(f"**评级:** {supplier['rating']}")
                        st.markdown(f"**合作内容:** {supplier['cooperation_content']}")
                    with col2:
                        st.markdown(f"**关联项目:** {supplier['project_relation']}")
                        st.markdown(f"**接触频率:** {supplier['contact_frequency']}")
                        st.markdown(f"**合同金额:** {supplier['contract_amount']}")
        else:
            st.info("未选择显示供应商信息或无供应商数据")

def display_family_network_graph(cadre_name, family_relations):
    """显示家属关系网络图"""
    import plotly.graph_objects as go
    import networkx as nx
    
    st.markdown("### 👨‍👩‍👧‍👦 家属关系网络图")
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加中心节点（干部本人）
    G.add_node(cadre_name, type='cadre', label=cadre_name, color='#e74c3c', size=25)
    
    # 添加父母节点
    if family_relations.get('father', {}).get('name'):
        father_name = family_relations['father']['name']
        G.add_node(f"father_{father_name}", type='father', label=father_name, 
                  color='#34495e', size=20)
        G.add_edge(cadre_name, f"father_{father_name}", relation="父子")
    
    if family_relations.get('mother', {}).get('name'):
        mother_name = family_relations['mother']['name']
        G.add_node(f"mother_{mother_name}", type='mother', label=mother_name, 
                  color='#9b59b6', size=20)
        G.add_edge(cadre_name, f"mother_{mother_name}", relation="母子")
    
    # 添加配偶节点
    if family_relations.get('spouse', {}).get('name'):
        spouse_name = family_relations['spouse']['name']
        G.add_node(f"spouse_{spouse_name}", type='spouse', label=spouse_name, 
                  color='#e67e22', size=22)
        G.add_edge(cadre_name, f"spouse_{spouse_name}", relation="夫妻")
    
    # 添加子女节点
    children = family_relations.get('children', [])
    for i, child in enumerate(children):
        child_name = child['name']
        G.add_node(f"child_{i}_{child_name}", type='child', label=child_name, 
                  color='#1abc9c', size=18)
        G.add_edge(cadre_name, f"child_{i}_{child_name}", relation="父子")
    
    # 使用圆形布局
    pos = nx.circular_layout(G)
    
    # 创建边
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='rgba(255,182,193,0.8)'),
        hoverinfo='none', mode='lines'
    )
    
    # 创建节点
    node_x, node_y, node_text, node_color, node_size, node_hover = [], [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_info = G.nodes[node]
        node_text.append(node_info['label'])
        node_color.append(node_info['color'])
        node_size.append(node_info['size'])
        
        # 悬浮信息
        node_type = node_info['type']
        if node_type == 'cadre':
            hover_text = f"<b>{node_info['label']}</b><br>身份: 本人"
        elif node_type == 'father':
            father_info = family_relations['father']
            hover_text = f"<b>{node_info['label']}</b><br>身份: 父亲<br>单位: {father_info.get('work_unit', '未知')}<br>职位: {father_info.get('job_position', '未知')}"
        elif node_type == 'mother':
            mother_info = family_relations['mother']
            hover_text = f"<b>{node_info['label']}</b><br>身份: 母亲<br>单位: {mother_info.get('work_unit', '未知')}<br>职位: {mother_info.get('job_position', '未知')}"
        elif node_type == 'spouse':
            spouse_info = family_relations['spouse']
            hover_text = f"<b>{node_info['label']}</b><br>身份: 配偶<br>单位: {spouse_info.get('work_unit', '未知')}<br>职位: {spouse_info.get('job_position', '未知')}"
        elif node_type == 'child':
            # 从节点ID中提取子女索引
            child_index = int(node.split('_')[1])
            child_info = children[child_index]
            hover_text = f"<b>{node_info['label']}</b><br>身份: 子女<br>单位: {child_info.get('work_unit', '未知')}<br>职位: {child_info.get('job_position', '未知')}"
        else:
            hover_text = f"<b>{node_info['label']}</b>"
        
        node_hover.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        hoverinfo='text', hovertext=node_hover,
        text=node_text, textposition="middle center",
        textfont=dict(size=12, color='white', family='Arial Black'),
        marker=dict(size=node_size, color=node_color,
                   line=dict(width=3, color='white'), opacity=0.9)
    )
    
    # 创建图表
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'👨‍👩‍👧‍👦 {cadre_name} 的家属关系网络',
            showlegend=False, hovermode='closest',
            margin=dict(b=60,l=20,r=20,t=60),
            plot_bgcolor='rgba(248,249,250,1)',
            annotations=[
                dict(
                    text="🔴 本人  ⚫ 父亲  🟣 母亲  🟠 配偶  🟢 子女<br>点击节点查看详情",
                    showarrow=False, xref="paper", yref="paper",
                    x=0.5, y=-0.08, xanchor='center', yanchor='bottom',
                    font=dict(size=12, color='#7f8c8d')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_family_details(family_relations):
    """显示家属详细信息"""
    st.markdown("### 👨‍👩‍👧‍👦 家属详细信息")
    
    # 使用标签页展示不同类型的家属
    tabs = []
    tab_contents = []
    
    # 父母信息
    if family_relations.get('father', {}).get('name') or family_relations.get('mother', {}).get('name'):
        tabs.append("👫 父母")
        tab_contents.append('parents')
    
    # 配偶信息
    if family_relations.get('spouse', {}).get('name'):
        tabs.append("💑 配偶")
        tab_contents.append('spouse')
    
    # 子女信息
    if family_relations.get('children'):
        tabs.append("👶 子女")
        tab_contents.append('children')
    
    if tabs:
        selected_tabs = st.tabs(tabs)
        
        for i, (tab, content_type) in enumerate(zip(selected_tabs, tab_contents)):
            with tab:
                if content_type == 'parents':
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 👨 父亲")
                        father = family_relations.get('father', {})
                        if father.get('name'):
                            st.markdown(f"**姓名:** {father['name']}")
                            st.markdown(f"**工作单位:** {father.get('work_unit', '未知')}")
                            st.markdown(f"**职位:** {father.get('job_position', '未知')}")
                        else:
                            st.info("无父亲信息")
                    
                    with col2:
                        st.markdown("#### 👩 母亲")
                        mother = family_relations.get('mother', {})
                        if mother.get('name'):
                            st.markdown(f"**姓名:** {mother['name']}")
                            st.markdown(f"**工作单位:** {mother.get('work_unit', '未知')}")
                            st.markdown(f"**职位:** {mother.get('job_position', '未知')}")
                        else:
                            st.info("无母亲信息")
                
                elif content_type == 'spouse':
                    st.markdown("#### 💑 配偶信息")
                    spouse = family_relations.get('spouse', {})
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**姓名:** {spouse['name']}")
                        st.markdown(f"**工作单位:** {spouse.get('work_unit', '未知')}")
                    with col2:
                        st.markdown(f"**职位:** {spouse.get('job_position', '未知')}")
                
                elif content_type == 'children':
                    st.markdown("#### 👶 子女信息")
                    children = family_relations.get('children', [])
                    for i, child in enumerate(children, 1):
                        with st.expander(f"👶 子女 {i}: {child['name']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**姓名:** {child['name']}")
                                st.markdown(f"**工作单位:** {child.get('work_unit', '未知')}")
                            with col2:
                                st.markdown(f"**职位:** {child.get('job_position', '未知')}")
    else:
        st.info("暂无家属信息")

def perform_integrity_risk_assessment(data_manager, cadre_name, risk_dimensions):
    """执行廉洁风险评估"""
    cadre = data_manager.get_cadre_by_name(cadre_name)
    risk_factors = []
    risk_score = 0
    
    # 工作关系风险
    if "工作关系风险" in risk_dimensions:
        projects = generate_work_projects_for_cadre(cadre)
        suppliers = generate_suppliers_for_cadre(cadre)
        
        high_value_projects = [p for p in projects if float(p['budget'].replace('万元', '').replace(',', '')) > 5000]
        if len(high_value_projects) > 3:
            risk_factors.append("参与多个高价值项目，存在利益输送风险")
            risk_score += 20
        
        a_level_suppliers = [s for s in suppliers if s['rating'] == 'A']
        if len(a_level_suppliers) > 5:
            risk_factors.append("与多个A级供应商密切合作，需关注利益关联")
            risk_score += 15
    
    # 家庭关系风险
    if "家庭关系风险" in risk_dimensions:
        if hasattr(cadre, 'family_relations') and cadre.family_relations:
            family = cadre.family_relations
            
            # 检查家属是否在相关行业工作
            risk_units = ["三峡", "电力", "建设", "能源", "电气"]
            for relation_type in ['father', 'mother', 'spouse']:
                if family.get(relation_type, {}).get('work_unit'):
                    work_unit = family[relation_type]['work_unit']
                    if any(unit in work_unit for unit in risk_units):
                        risk_factors.append(f"{relation_type}在相关行业工作，存在关联交易风险")
                        risk_score += 10
            
            for child in family.get('children', []):
                if child.get('work_unit'):
                    work_unit = child['work_unit']
                    if any(unit in work_unit for unit in risk_units):
                        risk_factors.append("子女在相关行业工作，存在关联交易风险")
                        risk_score += 10
    
    # 职位权力风险
    if "职位权力风险" in risk_dimensions:
        high_risk_positions = ["主管", "负责人", "经理", "总监", "主任"]
        if any(pos in cadre.position for pos in high_risk_positions):
            risk_factors.append("担任关键职位，权力集中，存在滥用风险")
            risk_score += 25
    
    # 历史行为风险
    if "历史行为风险" in risk_dimensions:
        # 模拟历史行为风险评估
        import random
        if random.random() < 0.3:  # 30%概率有历史风险
            risk_factors.append("历史行为记录中存在异常交易模式")
            risk_score += 30
    
    # 外部关联风险
    if "外部关联风险" in risk_dimensions:
        # 模拟外部关联风险
        import random
        if random.random() < 0.2:  # 20%概率有外部风险
            risk_factors.append("存在未申报的外部兼职或投资关联")
            risk_score += 35
    
    # 确定风险等级
    if risk_score <= 20:
        risk_level = "低风险"
    elif risk_score <= 50:
        risk_level = "中风险"
    else:
        risk_level = "高风险"
    
    # 生成预警信息和建议措施
    warnings = []
    suggestions = []
    
    if risk_level == "高风险":
        warnings.append("风险等级较高，建议立即开展专项调查")
        suggestions.extend([
            "立即开展廉洁谈话",
            "加强项目监督检查",
            "建立专项档案跟踪",
            "限制相关决策权限"
        ])
    elif risk_level == "中风险":
        warnings.append("存在一定风险，需加强日常监督")
        suggestions.extend([
            "定期开展廉洁提醒",
            "强化制度执行监督",
            "建立风险预警机制"
        ])
    else:
        suggestions.extend([
            "继续保持良好作风",
            "加强廉洁自律教育",
            "建立长效监督机制"
        ])
    
    return {
        "风险等级": risk_level,
        "风险分数": risk_score,
        "风险因素": risk_factors,
        "预警信息": warnings,
        "建议措施": suggestions
    }

def display_risk_assessment_results(risk_result):
    """显示风险评估结果"""
    st.markdown("### ⚠️ 风险评估结果")
    
    # 风险等级和分数
    col1, col2 = st.columns(2)
    
    risk_colors = {
        "低风险": "#4ECDC4",
        "中风险": "#FECA57", 
        "高风险": "#FF6B6B"
    }
    
    color = risk_colors.get(risk_result["风险等级"], "#667eea")
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, {color}88 100%); 
                   color: white; padding: 20px; border-radius: 15px; text-align: center;
                   box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 1.2em; margin-bottom: 5px;">⚠️</div>
            <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{risk_result["风险等级"]}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">风险等级</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 15px; text-align: center;
                   box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
            <div style="font-size: 1.2em; margin-bottom: 5px;">📊</div>
            <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">{risk_result["风险分数"]}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">风险分数</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 风险因素
    if risk_result["风险因素"]:
        st.markdown("#### 🔍 识别的风险因素")
        for i, factor in enumerate(risk_result["风险因素"], 1):
            st.markdown(f"""
            <div style="background: #fff3cd; border-left: 4px solid #ffc107; 
                       padding: 12px; margin: 8px 0; border-radius: 4px;">
                <strong>{i}.</strong> {factor}
            </div>
            """, unsafe_allow_html=True)
    
    # 预警信息
    if risk_result["预警信息"]:
        st.markdown("#### 📢 预警信息")
        for warning in risk_result["预警信息"]:
            st.markdown(f"""
            <div style="background: #f8d7da; border-left: 4px solid #dc3545; 
                       padding: 12px; margin: 8px 0; border-radius: 4px;">
                <strong>⚠️ 预警:</strong> {warning}
            </div>
            """, unsafe_allow_html=True)
    
    # 建议措施
    if risk_result["建议措施"]:
        st.markdown("#### 💡 建议措施")
        for i, suggestion in enumerate(risk_result["建议措施"], 1):
            st.markdown(f"""
            <div style="background: #d1ecf1; border-left: 4px solid #17a2b8; 
                       padding: 12px; margin: 8px 0; border-radius: 4px;">
                <strong>{i}.</strong> {suggestion}
            </div>
            """, unsafe_allow_html=True)

def display_overall_statistics(data_manager, cadre_names):
    """显示整体统计信息"""
    st.markdown("### 📈 整体统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 统计数据
    total_cadres = len(cadre_names)
    total_projects = 0
    total_suppliers = 0
    total_family_members = 0
    
    for name in cadre_names:
        cadre = data_manager.get_cadre_by_name(name)
        projects = generate_work_projects_for_cadre(cadre)
        suppliers = generate_suppliers_for_cadre(cadre)
        total_projects += len(projects)
        total_suppliers += len(suppliers)
        
        if hasattr(cadre, 'family_relations') and cadre.family_relations:
            family = cadre.family_relations
            total_family_members += 2  # 父母
            if family.get('spouse', {}).get('name'):
                total_family_members += 1
            total_family_members += len(family.get('children', []))
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 干部总数</h3>
            <h1 style="color: #667eea;">{total_cadres}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 项目总数</h3>
            <h1 style="color: #667eea;">{total_projects}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏢 供应商数</h3>
            <h1 style="color: #667eea;">{total_suppliers}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👨‍👩‍👧‍👦 家属数</h3>
            <h1 style="color: #667eea;">{total_family_members}</h1>
        </div>
        """, unsafe_allow_html=True)

def display_overall_network_graph(data_manager, cadre_names):
    """显示整体关系网络图"""
    st.markdown("### 🌐 整体关系网络")
    
    # 由于整体网络可能很复杂，提供简化视图
    st.info("整体关系网络图正在开发中，当前显示重点关系概览")
    
    # 可以显示一些重点关系的统计图表
    import plotly.express as px
    
    # 按部门统计干部数量
    dept_counts = {}
    for name in cadre_names:
        cadre = data_manager.get_cadre_by_name(name)
        dept = cadre.department
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    
    fig = px.bar(
        x=list(dept_counts.keys()),
        y=list(dept_counts.values()),
        title="各部门干部分布",
        labels={'x': '部门', 'y': '人数'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def display_risk_ranking(data_manager, cadre_names):
    """显示风险排行榜"""
    st.markdown("### 🏆 风险评估排行")
    
    # 为所有干部进行风险评估
    risk_data = []
    for name in cadre_names:
        risk_result = perform_integrity_risk_assessment(
            data_manager, name, 
            ["工作关系风险", "家庭关系风险", "职位权力风险"]
        )
        risk_data.append({
            "姓名": name,
            "风险等级": risk_result["风险等级"],
            "风险分数": risk_result["风险分数"],
            "风险因素数": len(risk_result["风险因素"])
        })
    
    # 按风险分数排序
    risk_data.sort(key=lambda x: x["风险分数"], reverse=True)
    
    # 显示前10名高风险干部
    st.markdown("#### ⚠️ 高风险干部排行（前10名）")
    
    for i, data in enumerate(risk_data[:10], 1):
        color = "#FF6B6B" if data["风险等级"] == "高风险" else "#FECA57" if data["风险等级"] == "中风险" else "#4ECDC4"
        
        st.markdown(f"""
        <div style="background: white; border-radius: 8px; padding: 15px; margin: 8px 0;
                   border-left: 4px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="font-size: 1.1em;">#{i} {data["姓名"]}</strong>
                    <span style="margin-left: 10px; padding: 4px 8px; background: {color}; 
                                color: white; border-radius: 12px; font-size: 0.8em;">
                        {data["风险等级"]}
                    </span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2em; font-weight: bold; color: {color};">
                        {data["风险分数"]}分
                    </div>
                    <div style="font-size: 0.9em; color: #666;">
                        {data["风险因素数"]}个风险因素
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# 新增辅助函数
def calculate_business_experience_match(cadre, position, scenario):
    """计算业务经验匹配度"""
    base_score = 60
    
    # 模拟项目经验分数
    if "项目经理" in position:
        base_score += min(cadre.work_years * 3, 25)
    elif "专员" in position:
        base_score += min(cadre.work_years * 2, 20)
    
    # 根据场景调整
    if "重大项目" in scenario and cadre.work_years >= 8:
        base_score += 15
    elif "紧急" in scenario and cadre.age <= 35:
        base_score += 10
    
    return min(base_score, 100)

def calculate_position_risk(cadre, position, scenario):
    """计算任职风险"""
    risk_score = 0
    
    # 年龄风险
    if cadre.age < 25 or cadre.age > 55:
        risk_score += 20
    
    # 经验风险
    if cadre.work_years < 3:
        risk_score += 25
    
    # 能力风险
    avg_competency = sum(cadre.river_competency.values()) / len(cadre.river_competency)
    if avg_competency < 65:
        risk_score += 30
    
    if risk_score >= 50:
        return "高风险"
    elif risk_score >= 25:
        return "中风险"
    else:
        return "低风险"

def estimate_adaptation_time(cadre, position, match_score):
    """估算适应周期"""
    if match_score >= 85:
        return "1-2周"
    elif match_score >= 70:
        return "1-2个月"
    elif match_score >= 60:
        return "3-6个月"
    else:
        return "6个月以上"

def analyze_business_fit(cadre, position, scenario):
    """分析业务适配性"""
    # 模拟相关项目经验
    related_projects = [
        f"三峡清洁能源项目 ({cadre.work_years//2}个)",
        f"长江大保护目 ({max(1, cadre.work_years//3)}个)",
        f"抽水蓄能工程 ({max(1, cadre.work_years//4)}个)"
    ]
    
    # 模拟技能匹配
    skill_match = {
        "招投标管理": "高" if cadre.river_competency.get("招投标管理能力", 70) >= 75 else "中",
        "项目管理": "高" if cadre.work_years >= 5 else "中",
        "风险控制": "高" if cadre.river_competency.get("风险管理能力", 70) >= 70 else "中"
    }
    
    return {
        "related_projects": related_projects,
        "skill_match": skill_match
    }

def create_comprehensive_radar_chart(cadre, position, matching_engine):
    """创建综合能力雷达图"""
    import plotly.graph_objects as go
    
    # 获取岗位要求
    requirements = matching_engine.position_requirements[position]
    core_abilities = requirements["核心能力"]
    
    # 模拟岗位标准分数
    position_scores = [85, 80, 90, 75, 85]  # 模拟岗位要求分数
    cadre_scores = [cadre.river_competency.get(ability, 70) for ability in core_abilities[:5]]
    
    fig = go.Figure()
    
    # 添加岗位要求
    fig.add_trace(go.Scatterpolar(
        r=position_scores,
        theta=core_abilities[:5],
        fill='toself',
        name='岗位要求',
        line=dict(color='rgba(255, 0, 0, 0.6)')
    ))
    
    # 添加个人能力
    fig.add_trace(go.Scatterpolar(
        r=cadre_scores,
        theta=core_abilities[:5],
        fill='toself',
        name=f'{cadre.name}能力',
        line=dict(color='rgba(0, 255, 0, 0.6)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="岗位要求 vs 个人能力对比"
    )
    
    return fig

def generate_intelligent_suggestions(cadre, position, scenario, match_result):
    """生成智能化建议"""
    suggestions = {
        "immediate_actions": [
            f"阅读{position}相关的最新行业规范和政策文件",
            f"与现任{position}进行深入交流，了解具体工作流程",
            "参加相关业务培训课程或研讨会"
        ],
        "30_day_plan": [
            f"完成{position}岗位所需的专业认证或考试",
            "参与1-2个实际项目，获得实操经验",
            "建立与关键利益相关者的工作关系"
        ],
        "90_day_goals": [
            f"成为{position}领域的专业人才，能够独立处理复杂问题",
            "在团队中发挥领导作用，带领项目取得突破性成果",
            "建立个人品牌和行业影响力"
        ]
    }
    
    return suggestions

def define_success_metrics(position, scenario):
    """定义成功指标"""
    base_metrics = {
        "业务指标": [
            "项目完成率达到95%以上",
            "成本控制在预算范围内",
            "客户满意度达到90%以上"
        ],
        "团队管理": [
            "团队成员能力提升显著",
            "团队合作效率提升5%以上",
            "人员流失率低于5%"
        ],
        "个人发展": [
            "获得相关专业认证或资质",
            "在行业会议或期刊发表观点",
            "成为公司内部培训师或专家"
        ]
    }
    
    return base_metrics

def analyze_position_competition(data_manager, position, matching_engine, current_cadre):
    """分析岗位竞争情况"""
    cadres = data_manager.get_all_cadres()
    competitors = []
    
    for cadre in cadres:
        if cadre.name != current_cadre:
            match_result = matching_engine.calculate_position_match(cadre, position)
            competitors.append((cadre.name, match_result["匹配度"]))
    
    # 排序竞争者
    competitors.sort(key=lambda x: x[1], reverse=True)
    top_competitors = competitors[:3]
    
    current_cadre_obj = data_manager.get_cadre_by_name(current_cadre)
    current_match = matching_engine.calculate_position_match(current_cadre_obj, position)
    current_score = current_match["匹配度"]
    
    advantages = []
    improvements = []
    
    if current_score >= top_competitors[0][1] if top_competitors else 0:
        advantages.append("综合匹配度在所有候选人中排名靠前")
    else:
        improvements.append(f"需要提升综合能力，目前落后于{top_competitors[0][0]}")
    
    return {
        "advantages": advantages if advantages else ["具备基本的岗位胜任能力"],
        "improvements": improvements if improvements else ["继续保持并提升优势"]
    }

def get_enhanced_position_details(position, scenario):
    """获取增强的岗位详情"""
    position_details = {
        "大水电项目经理": {
            "duties": [
                "负责大型水电项目的统筹规划和执行管理",
                "协调各专业部门，确保项目进度和质量",
                "管理项目预算和成本控制",
                "处理项目风险和突发问题"
            ],
            "challenges": [
                "项目复杂度高，涉及多个专业领域",
                "工期紧张，需要精准的进度控制",
                "需要平衡多方利益相关者的需求"
            ],
            "career_path": [
                "项目经理 → 项目总监 → 事业部副总经理",
                "技术专家路线：项目经理 → 首席工程师 → 技术总监"
            ],
            "salary_range": "25-40万元/年，根据项目规模调整"
        }
    }
    
    return position_details.get(position, {
        "duties": ["按照岗位说明书执行相关职责"],
        "challenges": ["需要不断学习和提升专业能力"],
        "career_path": ["根据个人能力和公司需要灵活发展"],
        "salary_range": "按照公司薪酬体系执行"
    })

# 人才发展辅助函数
def generate_capability_diagnosis(cadre, development_goal):
    """生成能力诊断报告"""
    sorted_competencies = sorted(cadre.river_competency.items(), key=lambda x: x[1], reverse=True)
    
    # 计算发展潜力
    potential_improvement = 0
    if "专家" in development_goal:
        potential_improvement = 15
    elif "管理" in development_goal:
        potential_improvement = 12
    elif "项目" in development_goal:
        potential_improvement = 10
    else:
        potential_improvement = 8
    
    return {
        "top_strength": sorted_competencies[0][0],
        "strength_score": sorted_competencies[0][1],
        "improvement_area": sorted_competencies[-1][0],
        "improvement_score": sorted_competencies[-1][1],
        "potential_level": "高潜力" if potential_improvement >= 12 else "中潜力",
        "potential_improvement": potential_improvement
    }

def generate_personalized_development_path(cadre, development_goal):
    """生成个性化发展路径"""
    if "专家" in development_goal:
        return {
            "short_term": {
                "actions": [
                    f"深入研究{cadre.department}的最新技术发展趋势",
                    "参与行业会议和学术论坛，发表专业观点",
                    "建立个人知识库和经验分享平台"
                ],
                "metrics": [
                    "每月发表至少1篇专业文章",
                    "参与至少52个行业交流活动",
                    "建立与10个行业专家的联系"
                ]
            },
            "medium_term": {
                "plans": [
                    "申请行业高级认证或资质",
                    "牵头重要技术创新项目",
                    "在公司内部建立技术专家形象"
                ],
                "outcomes": [
                    "成为公司技术顾问或内部培训师",
                    "在行业内具有一定的知名度和影响力",
                    "能够独立主导复杂技术项目的规划和实施"
                ]
            },
            "long_term": {
                "goals": [
                    "成为行业公认的技术专家和意见领袖",
                    "参与行业标准制定和政策制定",
                    "在国内外专业领域具有重要影响力"
                ],
                "growth_potential": [
                    "公司首席技术官 (CTO)",
                    "行业协会技术委员会主任",
                    "国家级专家库成员"
                ]
            }
        }
    else:
        # 其他发展目标的通用路径
        return {
            "short_term": {
                "actions": [
                    "完成当前岗位的核心技能提升",
                    "参与跨部门协作项目的实践",
                    "建立更广泛的内部工作网络"
                ],
                "metrics": [
                    "个人绩效评估达到优秀等级",
                    "负责项目的成功率达到95%以上",
                    "团队合作满意度达到90%以上"
                ]
            },
            "medium_term": {
                "plans": [
                    "承担更多的管理和领导责任",
                    "参与重要战略项目的决策过程",
                    "建立跨部门的影响力和协调能力"
                ],
                "outcomes": [
                    "成为部门或团队的骨干成员",
                    "具备独立管理复杂项目的能力",
                    "在公司内部具有良好的声誉和影响力"
                ]
            },
            "long_term": {
                "goals": [
                    "成为部门高级管理者或专业领导",
                    "在业务领域成为公司的核心人才",
                    "为公司的战略发展做出重要贡献"
                ],
                "growth_potential": [
                    "部门副总经理或总经理",
                    "公司高级管理层成员",
                    "行业已知的专业领导者"
                ]
            }
        }

def generate_precision_training_plan(cadre, development_goal):
    """生成精准化培训计划"""
    # 基础培训计划
    mandatory_courses = [
        {
            "name": "三峡物资招标业务深度解析",
            "duration": "3天",
            "priority": "高",
            "expected_outcome": "全面掌握招标业务流程和核心要求"
        },
        {
            "name": "风险管理与内控建设",
            "duration": "2天",
            "priority": "高",
            "expected_outcome": "提升风险识别和控制能力"
        },
        {
            "name": "数字化转型与智能管理",
            "duration": "1周",
            "priority": "中",
            "expected_outcome": "掌握数字化工具和管理方法"
        }
    ]
    
    optional_courses = [
        {
            "name": "高级项目管理认证 (PMP)",
            "recommendation_score": 85,
            "scenario": "管理能力提升"
        },
        {
            "name": "招投标法律实务高级研修",
            "recommendation_score": 78,
            "scenario": "专业能力深化"
        },
        {
            "name": "行业领导力与战略管理",
            "recommendation_score": 72,
            "scenario": "高级管理发展"
        }
    ]
    
    practical_projects = [
        {
            "name": "三峡清洁能源基地供应链优化项目",
            "type": "实际业务项目",
            "role": "项目副经理",
            "duration": "6个月",
            "skill_improvement": {
                "项目管理": 15,
                "供应链管理": 12,
                "团队协作": 10
            }
        },
        {
            "name": "数字化采购平台升级改造",
            "type": "技术创新项目",
            "role": "业务专家",
            "duration": "4个月",
            "skill_improvement": {
                "技术创新": 18,
                "流程优化": 15,
                "数据分析": 12
            }
        }
    ]
    
    return {
        "mandatory_courses": mandatory_courses,
        "optional_courses": optional_courses,
        "practical_projects": practical_projects
    }

def generate_career_prediction(cadre, development_goal):
    """生成职业发展预测"""
    import random
    
    base_success_score = 70
    if cadre.age <= 35:
        base_success_score += 10
    if cadre.work_years >= 8:
        base_success_score += 5
    
    avg_competency = sum(cadre.river_competency.values()) / len(cadre.river_competency)
    if avg_competency >= 80:
        base_success_score += 10
    
    return {
        "success_indicators": {
            "overall_score": min(base_success_score, 95)
        },
        "risk_assessment": {
            "level": "低" if base_success_score >= 80 else "中" if base_success_score >= 65 else "高"
        },
        "adaptation_capability": min(75 + (cadre.work_years * 2), 95),
        "near_term_milestones": [
            {"date": "2024年6月", "event": "完成核心能力认证", "success_probability": 85},
            {"date": "2024年12月", "event": "承担重要项目领导角色", "success_probability": 75}
        ],
        "long_term_milestones": [
            {"timeframe": "2-3年内", "achievement": "晚级管理岗位", "prerequisites": "管理经验+业务专业度"},
            {"timeframe": "5年内", "achievement": "部门高级管理者", "prerequisites": "卓越领导力+战略思维"}
        ]
    }

def create_career_trajectory_chart(cadre, prediction):
    """创建职业发展轨迹图"""
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    
    # 模拟数据
    years = [datetime.now().year + i for i in range(6)]
    capability_scores = [75, 78, 82, 85, 88, 90]  # 模拟能力发展轨迹
    position_levels = [1, 1.2, 1.8, 2.5, 3.2, 4.0]  # 模拟职位等级
    
    fig = go.Figure()
    
    # 添加能力发展曲线
    fig.add_trace(go.Scatter(
        x=years,
        y=capability_scores,
        mode='lines+markers',
        name='能力发展轨迹',
        line=dict(color='#3498DB', width=3),
        yaxis='y'
    ))
    
    # 添加职位发展曲线
    fig.add_trace(go.Scatter(
        x=years,
        y=position_levels,
        mode='lines+markers',
        name='职位发展轨迹',
        line=dict(color='#E74C3C', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'{cadre.name}的职业发展预测轨迹',
        xaxis=dict(title='年份'),
        yaxis=dict(
            title='能力分数',
            side='left'
        ),
        yaxis2=dict(
            title='职位等级',
            side='right',
            overlaying='y'
        ),
        hovermode='x unified'
    )
    
    return fig

if __name__ == "__main__":
    main()
