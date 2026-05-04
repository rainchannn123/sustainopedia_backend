import os
import random
import json
import numpy as np
import bw2io as bi
from bw2data import Database, databases, Method, methods, projects
from bw2io import create_default_biosphere3
from bw2io import SingleOutputEcospold2Importer
from bw2io.strategies import normalize_units, set_biosphere_type, drop_unspecified_subcategories, link_iterable_by_fields, drop_unlinked_cfs
from bw2io.data import convert_lcia_methods_data
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches
import logging
import openpyxl
import functools
import bw2calc as bc
import pandas as pd

from bw2io import ExcelImporter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ecoinvent_parse.log')
    ]
)

# ---------------------------------------------------------------------------
# Unit compatibility: map input unit strings → ecoinvent unit string(s).
# Activities whose unit is NOT in the normalised set will be excluded from
# the candidate pool before semantic scoring, preventing mass-flow queries
# from matching capital-goods / infrastructure "unit"-scale activities.
# ---------------------------------------------------------------------------
_UNIT_NORM_MAP: dict[str, list[str]] = {
    # mass
    "kilogram":       ["kilogram"],
    "kg":             ["kilogram"],
    "gram":           ["gram", "kilogram"],
    "ton":            ["kilogram", "ton"],
    "tonne":          ["kilogram", "ton"],
    "metric ton":     ["kilogram", "ton"],
    # energy
    "kilowatt hour":  ["kilowatt hour"],
    "kwh":            ["kilowatt hour"],
    "kwh":            ["kilowatt hour"],
    "megajoule":      ["megajoule"],
    "mj":             ["megajoule"],
    "gigajoule":      ["gigajoule"],
    "gj":             ["gigajoule"],
    # transport
    "ton kilometer":       ["ton kilometer"],
    "tonne kilometer":     ["ton kilometer"],
    "tkm":                 ["ton kilometer"],
    "ton-kilometer":       ["ton kilometer"],
    # volume
    "cubic meter":    ["cubic meter"],
    "m3":             ["cubic meter"],
    "m³":             ["cubic meter"],
    "litre":          ["litre"],
    "liter":          ["litre"],
    "liter":          ["litre"],
    # length / area
    "meter":          ["meter", "metre"],
    "metre":          ["meter", "metre"],
    "square meter":   ["square meter"],
    # capital goods / infrastructure
    "unit":           ["unit"],
    "piece":          ["unit"],
    "vehicle":        ["unit"],
}

def _normalize_unit(unit_str: str | None) -> list[str] | None:
    """Return the ecoinvent unit string(s) that correspond to *unit_str*, or None."""
    if not unit_str:
        return None
    return _UNIT_NORM_MAP.get(unit_str.lower().strip(), None)

# 转换 Activity 对象为字典
def node_to_dict(node):
    return {
        "name": node.get('name', ''),
        "unit": node.get('unit', ''),
        "location": node.get('location', None),
        "database": node.get('database', ''),
        "code": node.get('code', ''),
        "reference_product": node.get('reference product', None)
    }

def convert_result(result):
    if isinstance(result, dict):
        return {k: convert_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [convert_result(item) for item in result]
    elif hasattr(result, '__class__') and result.__class__.__name__ == "Activity":
        return node_to_dict(result)
    elif hasattr(result, 'get'):
        return node_to_dict(result)
    return result

class EcoinventEngine:
    def __init__(self, db=None, project_name="default_project"):
        self.project_name = project_name
        self.db = db
        self.model = SentenceTransformer('./_models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf', device='cpu')
        self.method = None
        self._setup_project()

    def _setup_project(self):
        if self.project_name not in projects:
            projects.create_project(self.project_name)
        projects.set_current(self.project_name)

    def find_process_exact(self, name, location=None):
        matches = [
            act for act in self.db
            if act['name'].lower() == name.lower() and (location is None or act['location'] == location)
        ]
        logger.warning(f"Exact match for '{name}' (location: {location}): {len(matches)} found")
        return matches

    def find_process_fuzzy(self, name, location=None, cutoff=0.6, n=5):
        all_names = [act['name'] for act in self.db]
        close = get_close_matches(name.lower(), [n.lower() for n in all_names], n=n, cutoff=cutoff)
        matches = [
            act for act in self.db
            if act['name'].lower() in [c.lower() for c in close] and (location is None or act['location'] == location)
        ]
        logger.warning(f"Fuzzy match for '{name}' (location: {location}): {len(matches)} found")
        return [node_to_dict(act) for act in matches]

    def find_process_semantic(self, name, location=None, top_k=20, min_score=0.2, unit=None):
        norm_units = _normalize_unit(unit)

        def _build_candidates(unit_filter):
            return [
                (f"{act['name']} {act.get('reference product', '')}".strip(), act)
                for act in self.db
                if (location is None or act['location'] == location)
                and (unit_filter is None or act.get('unit', '').lower() in unit_filter)
            ]

        if norm_units:
            unit_filter_lower = [u.lower() for u in norm_units]
            candidates = _build_candidates(unit_filter_lower)
            if not candidates:
                logger.warning(
                    f"No candidates with unit '{unit}' for '{name}' (location: {location}); "
                    "falling back to all units."
                )
                candidates = _build_candidates(None)
        else:
            candidates = _build_candidates(None)

        if not candidates:
            logger.warning(f"No candidates for '{name}' (location: {location})")
            return []
        names = [c[0] for c in candidates]
        name_emb = self.model.encode([name], convert_to_tensor=True)
        name_vecs = self.model.encode(names, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(name_emb, name_vecs)[0].cpu().numpy()
        top_matches = [
            (score, cand) for score, cand in zip(scores, candidates)
            if score >= min_score
        ]
        top_matches = sorted(top_matches, key=lambda x: -x[0])[:top_k]
        results = [
            {"activity": node_to_dict(match[1][1]), "score": float(match[0])}
            for match in top_matches
        ]
        logger.warning(
            f"Semantic match for '{name}' (location: {location}, unit_filter: {norm_units}): "
            f"{len(results)} found, top scores: {[r['score'] for r in results]}")
        return results

    def resolve_activity(self, item, strict_mode=False):
        name = item["name"] if "name" in item else item["activity_name"]
        location = item.get("location", None)
        unit = item.get("unit", None)
        result = {
            "input_name": name,
            "location": location,
            "stage": item.get("stage"),
            "matched_activity": None,
            "match_type": "not_found",
            "suggestions": [],
            "score": None
        }
        logger.warning(f"Resolving activity: '{name}' (location: {location}, unit: {unit})")

        # ── Step 1: Exact name match (case-insensitive) ───────────────────────
        # This is the highest-confidence match and must run first.  Semantic
        # embeddings can confuse structurally similar names (e.g. all "mine
        # construction, *" activities) so an exact hit must short-circuit.
        exact_matches = [
            act for act in self.db
            if act['name'].lower() == name.lower()
            and (location is None or act['location'] == location)
        ]
        if exact_matches:
            result["matched_activity"] = exact_matches[0]
            result["match_type"] = "exact"
            logger.warning(f"Exact match: '{exact_matches[0]['name']}' (code: {exact_matches[0]['code']})")
            return result

        # ── Step 2: Semantic search + hybrid re-rank ──────────────────────────
        # top_k=10 gives the re-ranker enough candidates to find the best one.
        semantic_matches = self.find_process_semantic(name, location, top_k=10, min_score=0.2, unit=unit)
        logger.warning(
            f"Semantic matches for '{name}': {len(semantic_matches)} found, "
            f"top scores: {[m['score'] for m in semantic_matches[:5]]}")

        if semantic_matches:
            # Hybrid re-ranking: blend cosine similarity with word-level
            # containment so that candidates sharing the query's specific
            # material/type words (e.g. "iron") rank above semantically close
            # but categorically different ones (e.g. "non-ferrous metal").
            import re as _re
            def _word_tokens(s: str) -> set:
                return set(_re.sub(r'[,\-/]', ' ', s.lower()).split())

            query_tokens = _word_tokens(name)

            def _hybrid_score(m: dict) -> float:
                cand_tokens = _word_tokens(m["activity"]["name"])
                if not query_tokens:
                    containment = 0.0
                else:
                    containment = len(query_tokens & cand_tokens) / len(query_tokens)
                return 0.6 * m["score"] + 0.4 * containment

            semantic_matches = sorted(semantic_matches, key=_hybrid_score, reverse=True)

            for match in semantic_matches:
                try:
                    matched_activity = self.db.get(match["activity"]["code"])
                    h_score = _hybrid_score(match)
                    result["matched_activity"] = matched_activity
                    result["match_type"] = "semantic"
                    result["score"] = match["score"]
                    result["suggestions"] = semantic_matches
                    logger.warning(
                        f"Hybrid-ranked match: '{matched_activity['name']}' "
                        f"(code: {matched_activity['code']}), "
                        f"sem={match['score']:.3f}, hybrid={h_score:.3f}")
                    return result
                except KeyError:
                    logger.warning(f"No Activity found for code: {match['activity']['code']}")

        logger.error(f"No activity matched for input: {item}")
        return result

    def compute_lcia_with_uncertainty(self, inputs, n_simulations: int = 25):
        results = []
        for input_item in inputs:
            resolution = self.resolve_activity(input_item)
            matched_activity = resolution["matched_activity"]
            if not matched_activity:
                logger.error(f"No activity matched for input: {input_item}")
                results.append({
                    "input": input_item,
                    "error": "No activity matched",
                    "impact": None,
                    "match_history": convert_result(resolution)
                })
                continue

            func_unit = {matched_activity: input_item.get("amount", 1.0)}
            logger.warning(
                f"Computing LCA for {matched_activity['name']} with method {self.method.name}", extra={"frontend": True})

            try:
                # ── Step 1: deterministic LCA for the mean score ──────────────────
                lca = bc.LCA(func_unit, method=self.method.name)
                lca.lci()

                inventory_sum = lca.inventory.sum()
                logger.info(f"LCI inventory sum: {inventory_sum}", extra={"frontend": True})
                if inventory_sum == 0:
                    logger.warning("LCI inventory is zero; check technosphere resolution or database import.", extra={"frontend": True})

                lca.lcia()
                det_score = lca.score
                logger.warning(f"Deterministic LCA score: {det_score}")

                # ── Step 2: Monte Carlo for uncertainty propagation ───────────────
                # MonteCarloLCA uses scipy cgs(atol='legacy') which is broken on
                # scipy >= 1.12. DirectSolvingMonteCarloLCA uses spsolve (LU) instead
                # and is both correct and stable.
                logger.warning(f"Starting Monte Carlo Simulation for {matched_activity['name']}...", extra={"frontend": True})
                if n_simulations > 1:
                    try:
                        mc_lca = bc.DirectSolvingMonteCarloLCA(func_unit, method=self.method.name)
                        scores = [next(mc_lca) for _ in range(n_simulations)]
                        logger.info(
                            f"DirectSolvingMonteCarloLCA ({n_simulations} runs): "
                            f"mean={np.mean(scores):.4g}, sd={np.std(scores):.4g}, "
                            f"p5={np.percentile(scores, 5):.4g}, p95={np.percentile(scores, 95):.4g}"
                        )
                    except Exception as mc_err:
                        import traceback as _tb
                        logger.error(
                            f"DirectSolvingMonteCarloLCA failed for '{matched_activity['name']}': {mc_err}\n"
                            + _tb.format_exc()
                        )
                        scores = [det_score]

                    if np.std(scores) == 0.0 and n_simulations > 1:
                        logger.warning(
                            f"MC std_dev is exactly 0 for '{matched_activity['name']}'. "
                            "This usually means the database has no uncertainty data encoded "
                            "(uncertainty_type=0 on all exchanges)."
                        )
                else:
                    scores = [det_score]

                impact_stats = {
                    "mean": float(np.mean(scores)),
                    "std_dev": float(np.std(scores)),
                    "percentile_5": float(np.percentile(scores, 5)),
                    "percentile_95": float(np.percentile(scores, 95)),
                    "unit": self.method.metadata.get("unit", "unit")
                }
                
                logger.info(f"LCIA computation for {matched_activity['name']} completed", extra={"frontend": True})

                results.append({
                "input": input_item,
                "match_type": resolution["match_type"],
                "matched_activity": node_to_dict(matched_activity),
                "impact": impact_stats,
                "match_history": convert_result(resolution)
                })

            except Exception as e:
             # 如果 LCA 计算本身失败（如方法不匹配），报错
                logger.error(f"LCA calculation failed for {matched_activity['name']}: {e}", extra={"frontend": True})
                results.append({
            "input": input_item,
            "error": str(e),
            "impact": None,
            "match_history": convert_result(resolution)
        })

        return {
            "goal": "Compare environmental impact with uncertainty",
            "method": list(self.method.name),
            "results": results
        }

    def from_json_and_compute(self, input_json: dict, uncertainty: bool = True, n_simulations: int = 100):
        method_key = tuple(input_json.get("method", []))
        logger.warning(f"Method key: {method_key}")

        self.method = Method(method_key)
        logger.warning(f"Number of method entries: {len(self.method.load())}")
        fu_list = []
        for fu in input_json.get("functional_unit", []):
            if "activity_name" not in fu:
                logger.error(f"Missing 'activity_name' in functional unit: {fu}")
                raise KeyError(f"Missing 'activity_name' in functional unit: {fu}")
            fu_entry = {
                "name": fu["activity_name"],
                "amount": fu["amount"],
                "unit": fu.get("unit", "kilogram"),
                "location": fu.get("location", None),
                "system_boundary": fu.get("system_boundary", "cradle-to-gate"),
                "stage": fu.get("stage", None)
            }
            fu_list.append(fu_entry)
            logger.debug(f"Functional unit: {fu_entry}")

        logger.warning(f"Functional units: {len(fu_list)} defined")
        # Always use compute_lcia_with_uncertainty, with n_simulations=1 for basic calculation
        return self.compute_lcia_with_uncertainty(fu_list, n_simulations=n_simulations if uncertainty else 1)

    def setup_database_and_methods(self, db_name: str, datasets_path: str = None):
        """Setup database and ensure it's accessible"""
        from bw2data import databases, Database
        from bw2io import create_default_biosphere3
        from bw2data import methods
        ef_methods = [m for m in methods]
        logger.info(str(ef_methods))

        # 1. 确保 biosphere3 存在
        if "biosphere3" not in databases:
            logger.warning("Creating biosphere3...")
            create_default_biosphere3()
            logger.warning("biosphere3 created.")
        else:
            logger.warning("biosphere3 already exists.")

        # 2. 检查数据库是否已存在
        if db_name in databases:
            logger.warning(f"Database {db_name} already exists, loading...")
            self.db = Database(db_name)
            logger.warning(f"Database {db_name} loaded successfully, activity count: {len(list(self.db))}")
            return self.db

        # 3. 如果数据库不存在且提供了数据集路径，则导入新数据库
        if not datasets_path:
            raise ValueError(f"Database {db_name} does not exist and no datasets_path provided")

        logger.warning(f"Database {db_name} does not exist, importing from {datasets_path}...")
        
        # 尝试导入数据库
        try:
            from init_database import init_databases
            init_result = init_databases()
            if init_result:
                logger.warning(f"Database {db_name} imported successfully")
                self.db = Database(db_name)
                return self.db
            else:
                raise ValueError(f"Failed to import database {db_name}")
        except Exception as e:
            logger.error(f"Error importing database: {e}")
            raise ValueError(f"Failed to import database {db_name}: {e}")

        return self.db

def main(json_file):
    with open(json_file, "r") as f:
        input_data = json.load(f)

    # 提供默认值
    project_name = input_data.get("project_name", "ecoinvent")
    dataset_name = input_data.get("dataset", "ecoinvent")
    
    logger.warning(f"使用项目名称: {project_name}")
    logger.warning(f"使用数据集: {dataset_name}")
    
    engine = EcoinventEngine(project_name=project_name)

    db = engine.setup_database_and_methods(
        db_name=dataset_name,
        datasets_path='./bw2_sources/Big Mushroom/Small Mushroom/ecoinvent_data'
    )

    result = engine.from_json_and_compute(input_data, uncertainty=True, n_simulations=25)

    converted_result = convert_result(result)

    # 基于输入文件名生成输出文件名
    import os
    input_filename = os.path.splitext(os.path.basename(json_file))[0]
    output_filename = f"{input_filename}_{project_name}_output.json"
    
    try:
        with open(output_filename, "w", encoding="utf-8") as json_file:
            json.dump(converted_result, json_file, indent=2)
        print(f"JSON 文件已成功写入：{output_filename}")
    except TypeError as e:
        print("TypeError:", e)
        print("Problematic object type:", type(converted_result))
    
    return converted_result

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
        logger.error(f"在文件夹 {folder_path} 中未找到JSON文件")
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
            original_cwd = '/app'
            output_parent = os.path.dirname(os.path.abspath(output_folder))
            if output_parent:
                os.chdir(output_parent)
            
            # 使用现有的main函数处理单个文件
            result = main(json_file)
            
            # 将生成的文件移动到输出文件夹
            input_filename = os.path.splitext(os.path.basename(json_file))[0]
            # 从结果中获取项目名称
            project_name = "ecoinvent"  # 默认项目名称
            if isinstance(result, dict) and "method" in result:
                project_name = result.get("method", ["ecoinvent"])[0] if result.get("method") else "ecoinvent"
            
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
            logger.error(f"处理文件失败 {json_file}: {e}")
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

            folder_path = sys.argv[2]
            output_folder = sys.argv[3] if len(sys.argv) > 3 else "results"
            process_multiple_json_files(folder_path, output_folder)
        else:

            main(sys.argv[1])
    else:
        # 默认处理test.json
        main('test.json')