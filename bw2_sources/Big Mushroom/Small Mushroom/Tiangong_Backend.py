import os
import glob
import xml.etree.ElementTree as ET
import traceback
from torch.backends.quantized import engine
from tqdm import tqdm
import logging
from collections import defaultdict
from bw2data import Database, projects, databases
from bw2data.backends.peewee import Activity, Exchange
from bw2io import create_default_biosphere3
import json
import datetime

logging.basicConfig(
    level=logging.INFO,  # 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tiangong_parse.log')
    ]
)
logger = logging.getLogger(__name__)

def node_to_dict(node):
    """将Activity对象转换为字典"""
    if hasattr(node, 'key'):
        return {
            'name': node.get('name', ''),
            'code': node.get('code', ''),
            'location': node.get('location', 'GLO'),
            'database': node.get('database', ''),
            'type': node.get('type', ''),
            'unit': node.get('unit', ''),
            'reference product': node.get('reference product', '')
        }
    return node


def convert_result(result):
    """转换结果为可序列化格式"""
    if isinstance(result, dict):
        return {k: convert_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [convert_result(item) for item in result]
    elif hasattr(result, '__class__') and result.__class__.__name__ == "Activity":
        return node_to_dict(result)
    elif hasattr(result, 'get'):
        return node_to_dict(result)
    return result


class TiangongEngine:
    def __init__(self, project_name="Tiangong"):
        self.project_name = project_name
        self.db = None
        self.method = None
        self.model = None
        self._setup_project()

    def _setup_project(self):
        """设置项目"""
        ######################################################################################################################################
        # del databases["Tiangong"]
        ######################################################################################################################################
        if self.project_name not in projects:
            projects.create_project(self.project_name)
        projects.set_current(self.project_name)
        logger.info(f"项目设置完成: {self.project_name}")
        logger.info(f"Project directory: {projects.dir}")
        logger.info(f"Current project: {projects.current}")

    def _setup_semantic_model(self):
        """设置语义匹配模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('./_models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf', device='cpu')
            logger.info("语义匹配模型加载成功")
        except Exception as e:
            logger.warning(f"语义匹配模型加载失败: {e}")
            self.model = None

    def setup_database(self):
        """设置Tiangong数据库"""
        logger.info(databases)
        if "Tiangong" in databases:
            self.db = Database("Tiangong")
            logger.info(f"使用现有Tiangong数据库，活动数量: {len(list(self.db))}")
        else:
            logger.warning("Tiangong数据库不存在，请先运行完整数据库建立")
            raise ValueError("Tiangong数据库不存在")
        return self.db

    def find_process_exact(self, name, location=None):
        """精确匹配活动"""
        matches = []
        for act in self.db:
            try:
                act_name = act.get('name', '')
                act_location = act.get('location', '')
                if (act_name.lower() == name.lower() and
                        (location is None or act_location == location)):
                    matches.append(act)
            except Exception as e:
                logger.warning(f"处理活动时出错: {e}")
                continue
        logger.info(f"精确匹配 '{name}' (location: {location}): 找到 {len(matches)} 个")
        return matches

    def find_process_fuzzy(self, name, location=None, cutoff=0.6, n=5):
        """模糊匹配活动"""
        from difflib import get_close_matches

        all_names = []
        for act in self.db:
            try:
                all_names.append(act.get('name', ''))
            except Exception as e:
                logger.warning(f"获取活动名称时出错: {e}")
                continue

        close = get_close_matches(name.lower(), [n.lower() for n in all_names], n=n, cutoff=cutoff)
        matches = []
        for act in self.db:
            try:
                act_name = act.get('name', '')
                act_location = act.get('location', '')
                if (act_name.lower() in [c.lower() for c in close] and
                        (location is None or act_location == location)):
                    matches.append(act)
            except Exception as e:
                logger.warning(f"处理活动时出错: {e}")
                continue
        logger.info(f"模糊匹配 '{name}' (location: {location}): 找到 {len(matches)} 个")
        return matches

    def resolve_activity(self, item):
        """解析活动，返回匹配结果"""
        name = item.get("name") or item.get("activity_name")
        location = item.get("location", None)
        unit = item.get("unit", None)

        result = {
            "input_name": name,
            "location": location,
            "unit": unit,
            "matched_activity": None,
            "match_type": "not_found",
            "suggestions": [],
            "score": None
        }

        logger.info(f"解析活动: '{name}' (location: {location}, unit: {unit})")
        logger.info(f"数据库活动数量: {len(list(self.db))}")

        # 1. 精确匹配
        exact_matches = self.find_process_exact(name, location)
        if exact_matches:
            result["matched_activity"] = exact_matches[0]
            result["match_type"] = "exact"
            logger.info(
                f"精确匹配: {exact_matches[0].get('name', 'Unknown')} (code: {exact_matches[0].get('code', 'Unknown')})")
            return result

        # 2. 模糊匹配
        fuzzy_matches = self.find_process_fuzzy(name, location, cutoff=0.6, n=5)
        if fuzzy_matches:
            result["matched_activity"] = fuzzy_matches[0]
            result["match_type"] = "fuzzy"
            result["suggestions"] = [{"name": act.get('name', 'Unknown'), "code": act.get('code', 'Unknown')} for act in
                                     fuzzy_matches]
            logger.info(
                f"模糊匹配: {fuzzy_matches[0].get('name', 'Unknown')} (code: {fuzzy_matches[0].get('code', 'Unknown')})")
            return result

        # 3. 关键词匹配（改进版，优先于语义匹配）
        keyword_matches = self.find_process_by_keywords(name, location, top_k=5)
        if keyword_matches:
            result["matched_activity"] = keyword_matches[0]["activity"]
            result["match_type"] = "keyword"
            result["score"] = keyword_matches[0]["score"]
            result["suggestions"] = keyword_matches
            logger.info(
                f"关键词匹配: {keyword_matches[0]['activity'].get('name', 'Unknown')} (code: {keyword_matches[0]['activity'].get('code', 'Unknown')}), score: {keyword_matches[0]['score']}")
            return result

        # 4. 语义匹配（如果可用）
        try:
            semantic_matches = self.find_process_semantic(name, location, top_k=20, min_score=0.2)
            if semantic_matches:
                result["matched_activity"] = semantic_matches[0]["activity"]
                result["match_type"] = "semantic"
                result["score"] = semantic_matches[0]["score"]
                result["suggestions"] = semantic_matches

                # 安全地获取活动信息用于日志
                activity = semantic_matches[0]["activity"]
                activity_name = activity.get('name', 'Unknown') if hasattr(activity, 'get') else getattr(activity,
                                                                                                         'name',
                                                                                                         'Unknown')
                activity_code = activity.get('code', 'Unknown') if hasattr(activity, 'get') else getattr(activity,
                                                                                                         'code',
                                                                                                         'Unknown')
                logger.info(f"语义匹配: {activity_name} (code: {activity_code}), score: {semantic_matches[0]['score']}")
                return result
        except Exception as e:
            logger.warning(f"语义匹配不可用: {e}")
            import traceback
            logger.warning(f"语义匹配错误详情: {traceback.format_exc()}")

        # 5. 如果仍然没有找到，尝试使用任何可用的活动作为回退
        fallback_activity = self.find_fallback_activity(name, location)
        if fallback_activity:
            result["matched_activity"] = fallback_activity
            result["match_type"] = "fallback"
            result["suggestions"] = [
                {"name": fallback_activity.get('name', 'Unknown'), "code": fallback_activity.get('code', 'Unknown')}]
            logger.info(
                f"使用回退活动: {fallback_activity.get('name', 'Unknown')} (code: {fallback_activity.get('code', 'Unknown')})")
            return result

        logger.info(f"未找到匹配的活动: {item}")
        return result

    def find_process_semantic(self, name, location=None, top_k=20, min_score=0.2):
        """语义匹配活动（使用SentenceTransformer）"""
        # 如果模型未加载，尝试加载
        if self.model is None:
            self._setup_semantic_model()

        if self.model is None:
            logger.info("语义匹配模型不可用，使用简化匹配")
            return self._find_process_simple_semantic(name, location, top_k, min_score)

        candidates = []
        for act in self.db:
            try:
                # 安全地获取位置信息
                act_location = act.get('location', '') if hasattr(act, 'get') else getattr(act, 'location', '')
                if location is None or act_location == location:
                    # 安全地构建候选文本
                    act_name = act.get('name', '') if hasattr(act, 'get') else getattr(act, 'name', '')
                    ref_product = act.get('reference product', '') if hasattr(act, 'get') else getattr(act,
                                                                                                       'reference product',
                                                                                                       '')
                    candidate_text = f"{act_name} {ref_product}".strip()
                    candidates.append((candidate_text, act))
            except Exception as e:
                logger.warning(f"处理活动时出错: {e}")
                continue

        if not candidates:
            logger.info(f"没有候选活动: '{name}' (location: {location})")
            return []

        try:
            from sentence_transformers import util

            # 准备文本
            names = [c[0] for c in candidates]
            name_emb = self.model.encode([name], convert_to_tensor=True)
            name_vecs = self.model.encode(names, convert_to_tensor=True)

            # 计算相似度
            scores = util.pytorch_cos_sim(name_emb, name_vecs)[0].cpu().numpy()

            # 筛选高分匹配
            top_matches = [
                (score, cand) for score, cand in zip(scores, candidates)
                if score >= min_score
            ]
            top_matches = sorted(top_matches, key=lambda x: -x[0])[:top_k]

            results = [
                {"activity": match[1], "score": float(match[0])}
                for match in top_matches
            ]

            logger.info(
                f"语义匹配 '{name}' (location: {location}): 找到 {len(results)} 个，最高分数: {[r['score'] for r in results[:3]]}")
            return results

        except Exception as e:
            logger.warning(f"语义匹配失败: {e}，使用简化匹配")
            return self._find_process_simple_semantic(name, location, top_k, min_score)

    def _find_process_simple_semantic(self, name, location=None, top_k=20, min_score=0.2):
        """简化的语义匹配（基于关键词重叠）"""
        candidates = []
        for act in self.db:
            try:
                # 安全地获取位置信息
                act_location = act.get('location', '') if hasattr(act, 'get') else getattr(act, 'location', '')
                if location is None or act_location == location:
                    # 安全地构建候选文本
                    act_name = act.get('name', '') if hasattr(act, 'get') else getattr(act, 'name', '')
                    ref_product = act.get('reference product', '') if hasattr(act, 'get') else getattr(act,
                                                                                                       'reference product',
                                                                                                       '')
                    candidate_text = f"{act_name} {ref_product}".strip()
                    candidates.append((candidate_text, act))
            except Exception as e:
                logger.warning(f"处理活动时出错: {e}")
                continue

        if not candidates:
            logger.info(f"没有候选活动: '{name}' (location: {location})")
            return []

        # 简化的相似度计算：基于关键词重叠
        name_words = set(name.lower().split())
        scored_candidates = []

        for candidate_text, act in candidates:
            candidate_words = set(candidate_text.lower().split())
            # 计算Jaccard相似度
            intersection = len(name_words & candidate_words)
            union = len(name_words | candidate_words)
            score = intersection / union if union > 0 else 0

            if score >= min_score:
                scored_candidates.append((score, {"activity": act, "score": score}))

        # 按分数排序
        scored_candidates.sort(key=lambda x: -x[0])
        results = [candidate[1] for candidate in scored_candidates[:top_k]]

        logger.info(
            f"简化语义匹配 '{name}' (location: {location}): 找到 {len(results)} 个，最高分数: {[r['score'] for r in results[:3]]}")
        return results

    def find_process_by_keywords(self, name, location=None, top_k=5):
        """基于关键词匹配活动（改进版，优先有生物圈交换的活动）"""
        name_words = set(name.lower().split())
        candidates = []

        # 特殊关键词映射
        keyword_mapping = {
            'printed': ['printing', 'printed'],
            'wiring': ['wiring', 'circuit', 'electronic'],
            'board': ['board', 'substrate', 'panel'],
            'battery': ['battery', 'lithium', 'lead'],
            'aluminium': ['aluminum', 'aluminium', 'foil'],
            'collector': ['collector', 'electrode'],
            'corrugated': ['corrugated', 'cardboard', 'box'],
            'injection': ['injection', 'molding', 'moulding'],
            'transport': ['transport', 'freight', 'lorry', 'truck'],
            'freight': ['freight', 'transport', 'lorry'],
            'lorry': ['lorry', 'truck', 'freight'],
            'euro6': ['euro6', 'euro', 'lorry', 'truck']
        }

        # 扩展搜索词
        expanded_words = set(name_words)
        for word in name_words:
            if word in keyword_mapping:
                expanded_words.update(keyword_mapping[word])

        for act in self.db:
            try:
                act_location = act.get('location', '') if hasattr(act, 'get') else getattr(act, 'location', '')
                if location is None or act_location == location:
                    act_name = act.get('name', '') if hasattr(act, 'get') else getattr(act, 'name', '')
                    act_words = set(act_name.lower().split())

                    # 计算关键词重叠（使用扩展词）
                    intersection = len(expanded_words & act_words)
                    if intersection > 0:
                        # 检查是否有生物圈交换
                        bio_count = 0
                        try:
                            for exc in act.exchanges():
                                if exc.get('type') == 'biosphere':
                                    bio_count += 1
                        except:
                            pass

                        # 基础分数
                        base_score = intersection / len(expanded_words)

                        # 如果有生物圈交换，大幅提升分数
                        if bio_count > 0:
                            # 生物圈交换数量越多，分数越高
                            bio_bonus = min(bio_count * 0.5, 2.0)  # 最多加2分
                            final_score = base_score + bio_bonus
                        else:
                            final_score = base_score

                        candidates.append((final_score, act, bio_count))
            except Exception as e:
                logger.warning(f"处理活动时出错: {e}")
                continue

        # 按分数排序
        candidates.sort(key=lambda x: -x[0])
        results = []
        for score, act, bio_count in candidates[:top_k]:
            results.append({
                "activity": act,
                "score": score
            })

        logger.info(f"关键词匹配 '{name}' (location: {location}): 找到 {len(results)} 个")
        if results:
            logger.info(
                f"最佳匹配: {results[0]['activity'].get('name', 'Unknown')} (分数: {results[0]['score']:.3f})")

        return results

    def find_fallback_activity(self, name, location=None):
        """查找回退活动（用于测试目的）"""
        # 尝试找到任何可用的活动
        for act in self.db:
            try:
                act_location = act.get('location', '') if hasattr(act, 'get') else getattr(act, 'location', '')
                if location is None or act_location == location:
                    # 检查是否有生产交换
                    has_production = False
                    try:
                        for exc in act.exchanges():
                            if exc.get('type') == 'production':
                                has_production = True
                                break
                    except:
                        pass

                    if has_production:
                        return act
            except Exception as e:
                logger.warning(f"处理活动时出错: {e}")
                continue

        # 如果没找到，返回第一个活动
        try:
            first_activity = next(iter(self.db))
            logger.info(f"使用第一个可用活动作为回退: {first_activity.get('name', 'Unknown')}")
            return first_activity
        except StopIteration:
            logger.warning("数据库中没有活动")
            return None

    def compute_lcia(self, inputs):
        """计算LCIA结果"""
        results = []

        for input_item in inputs:
            resolution = self.resolve_activity(input_item)
            matched_activity = resolution["matched_activity"]

            if not matched_activity:
                logger.info(f"未找到匹配的活动: {input_item}")
                results.append({
                    "input": input_item,
                    "error": "未找到匹配的活动",
                    "impact": None,
                    "match_history": convert_result(resolution)
                })
                continue

            # 安全地获取活动信息
            try:
                if hasattr(matched_activity, 'get'):
                    activity_name = matched_activity.get('name', 'Unknown')
                    activity_code = matched_activity.get('code', 'Unknown')
                elif isinstance(matched_activity, tuple):
                    activity_name = str(matched_activity[0]) if len(matched_activity) > 0 else 'Unknown'
                    activity_code = str(matched_activity[1]) if len(matched_activity) > 1 else 'Unknown'
                else:
                    activity_name = str(matched_activity)
                    activity_code = 'Unknown'
            except Exception as e:
                logger.warning(f"获取活动信息时出错: {e}")
                activity_name = 'Unknown'
                activity_code = 'Unknown'

            func_unit = {matched_activity: input_item.get("amount", 1.0)}
            logger.info(f"计算LCA: {activity_name} (code: {activity_code}) 方法: {self.method}")
            logger.info(f"功能单元: {func_unit}")
            logger.info(f"方法类型: {type(self.method)}")

            try:
                from bw2calc import LCA, LeastSquaresLCA

                # 确保在正确的项目上下文中
                from bw2data import projects
                projects.set_current(self.project_name)

                # 确保方法是正确的格式
                if hasattr(self.method, 'name'):
                    method_name = self.method.name
                elif isinstance(self.method, tuple):
                    method_name = self.method
                else:
                    method_name = str(self.method)

                logger.info(f"使用方法: {method_name}")
                logger.info(f"方法类型: {type(method_name)}")

                # 验证方法对象
                try:
                    if hasattr(self.method, 'metadata'):
                        logger.info(f"方法元数据: {self.method.metadata}")
                    if hasattr(self.method, 'load'):
                        logger.info("方法对象有load方法")
                except Exception as e:
                    logger.warning(f"验证方法对象时出错: {e}")

                # 首先尝试普通 LCA（避免 LeastSquaresLCA 的警告）
                try:
                    lca = LCA(func_unit, method_name)
                    lca.lci()
                    lca.lcia()
                    calc_type = "LCA"
                    logger.info(f"✓ 使用普通 LCA 成功计算")
                except Exception as e:
                    logger.warning(f"普通 LCA 失败: {e}，尝试 LeastSquaresLCA")
                    # 如果普通 LCA 失败，尝试 LeastSquaresLCA
                    try:
                        lca = LeastSquaresLCA(func_unit, method_name)
                        lca.lci()
                        lca.lcia()
                        calc_type = "LeastSquaresLCA"
                        logger.info(f"✓ 使用 LeastSquaresLCA 成功计算")
                    except Exception as e2:
                        logger.warning(f"LeastSquaresLCA 也失败: {e2}")
                        # 尝试使用字符串形式的方法名称
                        try:
                            if isinstance(method_name, tuple):
                                method_str = " - ".join(str(x) for x in method_name)
                            else:
                                method_str = str(method_name)
                            logger.warning(f"尝试使用字符串方法名称: {method_str}")
                            lca = LCA(func_unit, method_str)
                            lca.lci()
                            lca.lcia()
                            calc_type = "LCA_string"
                            logger.info(f"✓ 使用字符串方法名称成功计算")
                        except Exception as e3:
                            logger.warning(f"字符串方法名称也失败: {e3}")
                            raise e3

                # 检查结果是否有效
                if lca.score is None or (
                        hasattr(lca.score, '__float__') and float(lca.score) != float(lca.score)):  # 检查是否为NaN
                    logger.warning(f"LCA计算结果为NaN，可能由于矩阵奇异问题")
                    impact_stats = {
                        "mean": None,
                        "std_dev": None,
                        "percentile_5": None,
                        "percentile_95": None,
                        "unit": "unit",
                        "note": "结果为NaN，可能由于矩阵奇异或数据不完整"
                    }
                else:
                    # 为了与Ecoinvent格式一致，使用相同的统计格式
                    impact_stats = {
                        "mean": float(lca.score),
                        "std_dev": 0.0,  # Tiangong没有uncertainty分析，设为0
                        "percentile_5": float(lca.score),
                        "percentile_95": float(lca.score),
                        "unit": "unit"  # 可以根据方法获取实际单位
                    }

                results.append({
                    "input": input_item,
                    "match_type": resolution["match_type"],
                    "matched_activity": node_to_dict(matched_activity),
                    "impact": impact_stats,
                    "match_history": convert_result(resolution)
                })

                logger.info(f"LCA计算结果: {lca.score} (使用 {calc_type})")

            except Exception as e:
                logger.warning(f"LCA计算失败 {activity_name}: {e}")
                logger.warning(f"详细错误信息: {traceback.format_exc()}")
                results.append({
                    "input": input_item,
                    "error": str(e),
                    "impact": None,
                    "match_history": convert_result(resolution)
                })

        return {
            "goal": "Calculate environmental impact",
            "method": list(self.method.name) if hasattr(self.method, 'name') else (
                list(self.method) if isinstance(self.method, tuple) else [str(self.method)]),
            "results": results
        }

    def from_json_and_compute(self, input_json: dict):
        """从JSON输入计算LCA"""
        from bw2data import methods, Method
        # 设置方法
        method_key = tuple(input_json.get("method", []))
        logger.info(f"使用方法: {method_key}")
        all_methods = list(methods)
        logger.info(f"Total number of methods: {len(all_methods)}")
        logger.info(f"Preview of all tiangong methods: {all_methods[:]}")

        # 直接尝试使用指定的方法
        if method_key in methods: #new
            # try:
            self.method = Method(method_key)
            logger.info(f"✓ 成功设置方法: {method_key}")
            # except Exception as e:
            #     logger.warning(f"方法设置失败: {e}")
        else:
            logger.info(f"指定方法未注册: {method_key}. 切换到回退逻辑。")

            # 如果指定的方法失败，尝试查找Tiangong方法
            tiangong_methods = []

            for method_name in methods:
                try:
                    if isinstance(method_name, tuple) and len(method_name) >= 3:
                        if method_name[0] == "Tiangong":
                            tiangong_methods.append(method_name)
                except Exception as e:
                    logger.warning(f"处理方法名称时出错: {e}")
                    continue

            if tiangong_methods:
                logger.info(f"找到可用的Tiangong方法: {tiangong_methods[:3]}")
                try:
                    self.method = Method(tiangong_methods[0])  # 使用第一个可用方法
                    logger.info(f"✓ 使用第一个Tiangong方法: {tiangong_methods[0]}")
                except Exception as e:
                    logger.warning(f"设置Tiangong方法失败: {e}")
                    raise ValueError(f"无法设置方法: {e}")
            else:
                # 尝试使用Environmental Footprint方法
                try:
                    fallback_method = ('Environmental Footprint', 'Climate change - 2017')
                    self.method = Method(fallback_method)
                    logger.info(f"使用备用方法: {fallback_method}")
                except Exception as e:
                    logger.warning(f"备用方法也失败: {e}")
                    # 尝试使用任何可用的方法
                    try:
                        if methods:
                            first_method = list(methods)[0]
                            logger.info(f"尝试使用第一个可用方法: {first_method}")
                            self.method = Method(first_method)
                            logger.info(f"✓ 使用第一个可用方法: {first_method}")
                        else:
                            raise ValueError("没有找到任何可用的方法")
                    except Exception as e2:
                        logger.warning(f"所有方法都失败: {e2}")
                        # 尝试使用CML方法作为最后的回退
                        try:
                            cml_method = ('CML 2001', 'climate change', 'GWP100')
                            self.method = Method(cml_method)
                            logger.info(f"使用CML回退方法: {cml_method}")
                        except Exception as e3:
                            logger.warning(f"CML方法也失败: {e3}")
                            raise ValueError(f"无法找到任何可用的方法: {e3}")

        # 验证方法对象
        try:
            if hasattr(self.method, 'name'):
                logger.info(f"方法名称: {self.method.name}")
            if hasattr(self.method, 'metadata'):
                logger.info(f"方法元数据: {self.method.metadata}")
        except Exception as e:
            logger.warning(f"验证方法对象时出错: {e}")

        # 解析功能单元
        fu_list = []
        for fu in input_json.get("functional_unit", []):
            if "activity_name" not in fu:
                logger.warning(f"功能单元缺少 'activity_name': {fu}")
                raise KeyError(f"功能单元缺少 'activity_name': {fu}")

            fu_entry = {
                "name": fu["activity_name"],
                "amount": fu["amount"],
                "unit": fu.get("unit", "kilogram"),
                "location": fu.get("location", "GLO"),
                "system_boundary": fu.get("system_boundary", "cradle-to-gate"),
                "stage": fu.get("stage", None)
            }
            fu_list.append(fu_entry)
            logger.info(f"功能单元: {fu_entry}")

        logger.info(f"功能单元数量: {len(fu_list)}")
        return self.compute_lcia(fu_list)


def list_all_methods():
    """列出所有可用的LCA方法"""
    try:
        from bw2data import methods, projects, databases
        print("=== 所有可用的LCA方法 ===")
        
        # 检查当前项目
        current_project = projects.current
        print(f"当前项目: {current_project}")
        
        # 检查Tiangong数据库是否存在
        if "Tiangong" in databases:
            print("✓ Tiangong数据库存在")
            tiangong_db = databases["Tiangong"]
            print(f"Tiangong数据库活动数量: {len(list(tiangong_db))}")
        else:
            print("✗ Tiangong数据库不存在")
        
        # 尝试切换到Tiangong项目
        print("\n=== 尝试切换到Tiangong项目 ===")
        tiangong_method_list = []
        ######################################################################################################################################
        # del databases["Tiangong"]
        ######################################################################################################################################
        try:
            if "Tiangong" not in projects:
                projects.create_project("Tiangong")
                print("✓ 创建Tiangong项目")
            logger.info(f"Project directory: {projects.dir}")
            logger.info(f"Current project: {projects.current}")
            projects.set_current("Tiangong")
            print("✓ 切换到Tiangong项目")
            
            # 重新检查方法
            from bw2data import methods as tiangong_methods
            print(f"Tiangong项目中的方法数量: {len(tiangong_methods)}")
            
            # 检查Tiangong方法
            tiangong_method_count = 0
            for method_name in tiangong_methods:
                if isinstance(method_name, tuple) and len(method_name) >= 1:
                    if method_name[0] == "Tiangong":
                        tiangong_method_count += 1
                        tiangong_method_list.append(method_name)
                        print(f"Tiangong方法 {tiangong_method_count}: {method_name}")
            
            if tiangong_method_count == 0:
                print("在Tiangong项目中仍未找到Tiangong方法")
            else:
                print(f"在Tiangong项目中找到 {tiangong_method_count} 个Tiangong方法")
                
                # 保存到txt文件
                with open("tiangong_methods.txt", "w", encoding="utf-8") as f:
                    f.write("=== Tiangong LCA方法列表 ===\n")
                    f.write(f"总数量: {tiangong_method_count} 个方法\n")
                    f.write("生成时间: " + str(datetime.datetime.now()) + "\n\n")
                    
                    for i, method in enumerate(tiangong_method_list, 1):
                        f.write(f"{i:2d}. {method}\n")
                    
                    f.write(f"\n=== 方法详情 ===\n")
                    for i, method in enumerate(tiangong_method_list, 1):
                        f.write(f"\n{i:2d}. {method}\n")
                        f.write(f"    数据库: {method[0]}\n")
                        f.write(f"    类别: {method[1]}\n")
                        f.write(f"    指标: {method[2]}\n")
                
                print(f"✓ Tiangong方法已保存到: tiangong_methods.txt")
                
        except Exception as e:
            print(f"切换到Tiangong项目失败: {e}")
        
        # 恢复原始项目
        try:
            projects.set_current(current_project)
            print(f"✓ 恢复到原始项目: {current_project}")
        except:
            pass
        
        # 按数据库分类统计
        method_stats = {}
        tiangong_methods = []
        other_methods = []
        
        print(f"\n正在检查 {len(methods)} 个方法...")
        
        for i, method_name in enumerate(methods):
            try:
                if isinstance(method_name, tuple) and len(method_name) >= 1:
                    database = method_name[0]
                    if database not in method_stats:
                        method_stats[database] = 0
                    method_stats[database] += 1
                    
                    # 分类Tiangong方法和其他方法
                    if database == "Tiangong":
                        tiangong_methods.append(method_name)
                        print(f"找到Tiangong方法: {method_name}")
                    else:
                        other_methods.append(method_name)
                        
                    # 打印前几个方法作为示例
                    if i < 5:
                        print(f"示例方法 {i+1}: {method_name}")
                        
            except Exception as e:
                print(f"警告: 处理方法名称时出错: {e}")
                continue
        
        # 打印统计信息
        print(f"\n总方法数量: {len(methods)}")
        print("按数据库分类:")
        for db, count in sorted(method_stats.items()):
            print(f"  {db}: {count} 个方法")
        
        # 打印Tiangong方法（前10个）
        print(f"\n=== Tiangong方法 (共{len(tiangong_methods)}个) ===")
        for i, method in enumerate(tiangong_methods[:10]):
            print(f"  {i+1}. {method}")
        if len(tiangong_methods) > 10:
            print(f"  ... 还有 {len(tiangong_methods) - 10} 个Tiangong方法")
        
        # 打印其他方法（前10个）
        print(f"\n=== 其他方法 (共{len(other_methods)}个) ===")
        for i, method in enumerate(other_methods[:10]):
            print(f"  {i+1}. {method}")
        if len(other_methods) > 10:
            print(f"  ... 还有 {len(other_methods) - 10} 个其他方法")
        
        # 特别检查Tiangong方法
        print(f"\n=== 详细检查Tiangong方法 ===")
        tiangong_count = 0
        for method_name in methods:
            if isinstance(method_name, tuple) and len(method_name) >= 1:
                if method_name[0] == "Tiangong":
                    tiangong_count += 1
                    print(f"Tiangong方法 {tiangong_count}: {method_name}")
        
        if tiangong_count == 0:
            print("未找到Tiangong方法")
        
        return {
            "total_methods": len(methods),
            "tiangong_methods": len(tiangong_methods),
            "other_methods": len(other_methods),
            "method_stats": method_stats,
            "current_project": current_project,
            "tiangong_db_exists": "Tiangong" in databases,
            "tiangong_method_list": tiangong_method_list
        }
        
    except Exception as e:
        print(f"错误: 获取方法列表失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None


def main(json_file_path):
    """从JSON文件运行Tiangong LCA计算"""

    try:
        # 读取JSON文件
        with open(json_file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # 创建TiangongEngine实例
        project_name = input_data.get("project_name", "Tiangong")
        engine = TiangongEngine(project_name=project_name)

        # 设置数据库
        db = engine.setup_database()

        # 计算LCA
        result = engine.from_json_and_compute(input_data)

        # 输出结果
        logger.info("=== LCA计算结果 ===")
        for res in result["results"]:
            if "error" in res:
                logger.info(f"错误: {res['input']['name']} - {res['error']}")
            else:
                impact_value = res['impact'].get('mean', 'Unknown')
                logger.info(f"✓ {res['matched_activity']['name']}: {impact_value}")

        # 转换为可序列化的格式
        converted_result = convert_result(result)

        # 基于输入文件名生成输出文件名
        import os
        input_filename = os.path.splitext(os.path.basename(json_file_path))[0]
        output_filename = f"{input_filename}_{project_name}_output.json"
        
        try:
            with open(output_filename, "w", encoding="utf-8") as json_file:
                json.dump(converted_result, json_file, indent=2)
            print(f"JSON 文件已成功写入：{output_filename}")
        except TypeError as e:
            print("TypeError:", e)
            print("Problematic object type:", type(converted_result))
        
        return converted_result

    except Exception as e:
        logger.warning(f"JSON测试失败: {e}")
        logger.warning(f"详细错误: {traceback.format_exc()}")
        raise


def process_multiple_json_files(folder_path, output_folder="results"):
    """处理文件夹中的多个JSON文件"""
    import os
    import glob
    import shutil
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.warning(f"创建输出文件夹: {output_folder}")
    
    # 查找所有JSON文件
    json_pattern = os.path.join(folder_path, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        logger.warning(f"在文件夹 {folder_path} 中未找到JSON文件")
        return
    
    logger.warning(f"找到 {len(json_files)} 个JSON文件")
    
    results_summary = {
        "processed_files": [],
        "successful_files": [],
        "failed_files": [],
        "total_files": len(json_files)
    }
    
    # 处理每个JSON文件
    for json_file in json_files:
        try:
            logger.warning(f"处理文件: {json_file}")
            
            # 临时切换到输出文件夹的父目录，让main函数在当前目录生成文件
            original_cwd = os.getcwd()
            output_parent = os.path.dirname(os.path.abspath(output_folder))
            if output_parent:
                os.chdir(output_parent)
            
            # 使用现有的main函数处理单个文件
            result = main(json_file)
            
            # 将生成的文件移动到输出文件夹
            input_filename = os.path.splitext(os.path.basename(json_file))[0]
            project_name = result.get("method", ["Tiangong"])[0] if result.get("method") else "Tiangong"
            generated_file = f"{input_filename}_{project_name}_output.json"
            
            if os.path.exists(generated_file):
                target_file = os.path.join(output_folder, generated_file)
                shutil.move(generated_file, target_file)
                logger.warning(f"输出文件已保存到: {target_file}")
            
            # 恢复原始工作目录
            os.chdir(original_cwd)
            
            # 记录成功处理
            results_summary["successful_files"].append({
                "file": json_file,
                "output_file": os.path.join(output_folder, generated_file) if os.path.exists(os.path.join(output_folder, generated_file)) else None,
                "result": result
            })
            
            logger.warning(f"✓ 成功处理: {json_file}")
            
        except Exception as e:
            logger.warning(f"处理文件失败 {json_file}: {e}")
            results_summary["failed_files"].append({
                "file": json_file,
                "error": str(e)
            })
    
    # 保存汇总结果
    summary_file = os.path.join(output_folder, "processing_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    logger.warning(f"处理完成。成功: {len(results_summary['successful_files'])}, 失败: {len(results_summary['failed_files'])}")
    logger.warning(f"汇总结果已保存到: {summary_file}")
    
    return results_summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--folder" and len(sys.argv) > 2:
            # 处理文件夹模式
            folder_path = sys.argv[2]
            output_folder = sys.argv[3] if len(sys.argv) > 3 else "results"
            process_multiple_json_files(folder_path, output_folder)
        elif sys.argv[1] == "--list-methods":
            # 列出所有方法
            list_all_methods()
        else:
            # 处理单个文件模式
            main(sys.argv[1])
    else:
        # 默认处理test.json
        main('test.json')

