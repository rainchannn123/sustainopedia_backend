# 根据环境变量决定导入
import os
import re
ENGINE_TYPE = os.getenv("ENGINE_TYPE", "ecoinvent")

# 根据环境变量设置可用性
if ENGINE_TYPE == "ecoinvent":
    try:
        from Ecoinvent_Backend import EcoinventEngine
        ECOINVENT_AVAILABLE = True
        TIANGONG_AVAILABLE = False  # 在 ecoinvent 容器中，Tiangong 始终不可用
    except ImportError:
        ECOINVENT_AVAILABLE = False
        TIANGONG_AVAILABLE = False
elif ENGINE_TYPE == "tiangong":
    try:
        from Tiangong_Backend import TiangongEngine
        TIANGONG_AVAILABLE = True
        ECOINVENT_AVAILABLE = False  # 在 tiangong 容器中，Ecoinvent 始终不可用
    except ImportError:
        TIANGONG_AVAILABLE = False
        ECOINVENT_AVAILABLE = False



import random
import json
import numpy as np
import logging
import traceback
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches
import functools
 

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Common utility functions
def node_to_dict(node):
    """Convert Activity object to dictionary"""
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
    """Convert result to serializable format"""
    if isinstance(result, dict):
        return {k: convert_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [convert_result(item) for item in result]
    elif hasattr(result, '__class__') and result.__class__.__name__ == "Activity":
        return node_to_dict(result)
    elif hasattr(result, 'get'):
        return node_to_dict(result)
    return result

def normalize_pattern(text):
    # Regex: find 'word' followed by ( ... )
    pattern = r"'([^']+)'\s*\(([^)]+)\)"
    replacement = r"'\1 (\2)'"
    return re.sub(pattern, replacement, text)


class UnifiedLCAEngine:
    """Unified LCA engine that can switch between Ecoinvent and Tiangong engines"""
    
    def __init__(self):
        self.ecoinvent_available = ECOINVENT_AVAILABLE
        self.tiangong_available = TIANGONG_AVAILABLE
        
        if not self.ecoinvent_available and not self.tiangong_available:
            raise ValueError("Neither Ecoinvent nor Tiangong dependencies are available")

    def create_engine(self, project_name: str):
        """Create appropriate engine based on project name and environment"""
        if ENGINE_TYPE == "tiangong":
            if not TIANGONG_AVAILABLE:
                raise ValueError("Tiangong engine not available in this container")
            logger.info(f"Creating Tiangong engine for project: {project_name}")
            return TiangongEngine(project_name=project_name)
        else:  # ENGINE_TYPE == "ecoinvent"
            if not ECOINVENT_AVAILABLE:
                raise ValueError("Ecoinvent engine not available in this container")
            logger.info(f"Creating Ecoinvent engine for project: {project_name}")
            return EcoinventEngineAdapter(EcoinventEngine(project_name=project_name))


        # 修改 run_lca_calculation 方法中的类型检查
    def run_lca_calculation(self, json_file_path: str):
        """Run LCA calculation based on JSON input"""
        print("SSSSSSSSSSSSTAAATTTTINGGGGGGGGGG LCA CALCULATIONNNNNNN")
        try:
            # Read JSON file
            with open(json_file_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)

            # Get project name and create appropriate engine
            project_name = "ecoinvent" if ENGINE_TYPE == "ecoinvent" else input_data.get("Project", "default_project")
            engine = self.create_engine(project_name)

            #Setup database based on engine type
            if ENGINE_TYPE == "tiangong" and TIANGONG_AVAILABLE:
                logger.info(f"Setting up Tiangong database for project: {project_name}")
                db = engine.setup_database()
            else:  # EcoinventEngineAdapter
                logger.info(f"Setting up Ecoinvent database for project: {project_name}")
                logger.info(f"Setting up Ecoinvent database for project...", extra={"frontend": True})
                db = engine.setup_database_and_methods(
                    db_name=input_data.get('dataset', 'ecoinvent'),
                    datasets_path=input_data.get('datasets_path', './bw2_sources/Big Mushroom/Small Mushroom/ecoinvent_data'))
                
                # Ensure the database is accessible on the adapter and engine
                logger.info(f"Finish Setting up Ecoinvent database for project: {project_name}")
                logger.info(f"Finish Setting up Ecoinvent database for project ✔", extra={"frontend": True})
                engine.db = db
                engine.engine.db = db

            # Compute LCA
            if ENGINE_TYPE == "tiangong" and TIANGONG_AVAILABLE:
                logger.warning("Using Tiangong engine for LCA calculation")
                result = engine.from_json_and_compute(input_data)
                logger.warning(f"LCA TIANGONG RESULTS: {result}")
            else:  # EcoinventEngineAdapter
                logger.info("Using Ecoinvent engine for LCA calculation")
                logger.info("Starting LCA calculation with Ecoinvent engine...", extra={"frontend": True})
                uncertainty = input_data.get('uncertainty', True)
                n_simulations = input_data.get('n_simulations', 25)  # 25 runs ≈ 80% precision of 100, ~4× faster
                result = engine.from_json_and_compute(input_data, uncertainty=uncertainty, n_simulations=n_simulations)
                logger.info("LCA calculation completed", extra={"frontend": True})

            # Convert result to serializable format
            import ast
            check_result = normalize_pattern(str(result))
            json_result = ast.literal_eval(check_result)
            converted_result = convert_result(json_result)
            # Generate output filename based on input filename
            import os
            input_filename = str(os.path.splitext(os.path.basename(json_file_path))[0])
            logger.warning(input_filename)
            output_filename = os.path.join(
                f"bw2_results/{input_filename}_output.json"
            )
            # Ensure output directory exists
            output_dir = os.path.dirname(output_filename) or "."
            try:
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                logger.warning(f"Could not create output directory {output_dir}: {e}")
            logger.warning("ready to write output bw2 file")
            # # Save result
            # logger.debug("Inspecting converted_result before JSON dump:")
            # # Recursively check for NaN/inf or non-serializable types
            # def check_for_issues(data, path=""):
            #     if isinstance(data, dict):
            #         for k, v in data.items():
            #             check_for_issues(v, f"{path}.{k}" if path else k)
            #     elif isinstance(data, list):
            #         for i, v in enumerate(data):
            #             check_for_issues(v, f"{path}[{i}]" if path else f"[{i}]")
            #     elif isinstance(data, float):
            #         if data != data:  # NaN check
            #             logger.warning(f"NaN found at {path}")
            #         elif data == float('inf'):
            #             logger.warning(f"Infinity found at {path}")
            #         elif data == float('-inf'):
            #             logger.warning(f"-Infinity found at {path}")
            #     # Add checks for other types if needed (e.g., custom objects)

            # check_for_issues(converted_result)
            
            # # Save result
            # try:
            #     # First, try to serialize to string for validation
            #     json_string = json.dumps(converted_result, indent=2, allow_nan=False)  # Set allow_nan=False to catch issues early
                
            #     # Write to file
            #     with open(output_filename, "w", encoding="utf-8") as json_file:
            #         json_file.write(json_string)
            #     print(f"JSON file successfully written: {output_filename}")
                
            #     # Verify by loading back immediately
            #     with open(output_filename, "r", encoding="utf-8") as json_file:
            #         loaded_data = json.load(json_file)
            #     logger.info("JSON file validated successfully after writing.")
            
            # except json.JSONEncodeError as e:
            #     logger.error(f"JSON encoding error: {e}")
            #     # Print problematic data snippet
            #     logger.error(f"Problematic data at: {e.msg} (line {e.lineno}, col {e.colno})")
            #     raise
            # except FileNotFoundError:
            #     print(f"Error: The bw2 output file could not be found or created.")
            # except TypeError as e:
            #     print(f"Error: A TypeError occurred during JSON serialization: {e}")
            #     # This might happen if data_to_write contains non-serializable objects
            # except IOError as e:
            #     print(f"Error: An I/O error occurred while writing to bw2 output: {e}")
            # except Exception as e:
            #     print(f"An unexpected error occurred: {e}")
            
 
    
            try:
                with open("debug_output.txt", "w", encoding="utf-8") as txt_file:
                    txt_file.write(str(converted_result))
                with open(output_filename, "w", encoding="utf-8") as json_file:
                    json.dump(converted_result, json_file, indent=2)
                logger.warning(f"JSON file successfully written: {output_filename}")
            except FileNotFoundError:
                logger.warning(f"Error: The bw2 output file could not be found or created.")
            except TypeError as e:
                logger.warning(f"Error: A TypeError occurred during JSON serialization: {e}")
                # This might happen if data_to_write contains non-serializable objects
            except IOError as e:
                logger.warning(f"Error: An I/O error occurred while writing to bw2 output: {e}")
            except Exception as e:
                logger.warning(f"An unexpected error occurred: {e}")

            return converted_result

        except Exception as e:
            logger.error(f"LCA calculation failed: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise

class EcoinventEngineAdapter:
    """Adapter class to add missing methods to EcoinventEngine without modifying the original"""
    
    def __init__(self, ecoinvent_engine):
        self.engine = ecoinvent_engine
        # Copy all attributes from the original engine except from_json_and_compute and setup_database_and_methods
        for attr in dir(ecoinvent_engine):
            if not attr.startswith('_') and attr not in ['from_json_and_compute', 'setup_database_and_methods']:
                setattr(self, attr, getattr(ecoinvent_engine, attr))
    
    def setup_database_and_methods(self, db_name: str, datasets_path: str = None):
        
        """Setup database and ensure it's accessible"""
        logger.info(f"EcoinventEngineAdapter.setup_database_and_methods called with db_name: {db_name}")

        # Ensure project is set up before database setup
        if hasattr(self, 'project_name'):
            logger.info(f"Setting up project: {self.project_name}")
            from bw2data import projects
            if self.project_name not in projects:
                projects.create_project(self.project_name)
            projects.set_current(self.project_name)
            logger.info(f"Project {self.project_name} is now current")
        
        # Call the original engine's method directly to avoid recursion
        db = self.engine.setup_database_and_methods(db_name, datasets_path)
        logger.info(f"Database returned from engine: {db}")
        # Ensure the database is set on both the engine and the adapter
        self.engine.db = db
        self.db = db
        logger.info(f"Database set on adapter: {self.db}")
        logger.info(f"Database set on engine: {self.engine.db}")
        return db
    
    def compute_lcia(self, inputs):
        """Compute basic LCIA for Ecoinvent - adapter method"""
        # Call the existing compute_lcia_with_uncertainty with n_simulations=1 for basic calculation
        return self.engine.compute_lcia_with_uncertainty(inputs, n_simulations=1)
    
    def from_json_and_compute(self, input_json: dict, uncertainty: bool = False, n_simulations: int = 100):
        """Override from_json_and_compute to handle the missing compute_lcia method"""
        logger.info("EcoinventEngineAdapter.from_json_and_compute called")
        method_key = tuple(input_json.get("method", []))
        logger.info(f"Method key: {method_key}")

        # Import Method from bw2data like in the original implementation
        from bw2data import Method
        self.engine.method = Method(method_key)
        
        # Ensure the database is set on the engine
        if hasattr(self, 'db') and self.db is not None:
            self.engine.db = self.db
            logger.info(f"Database set on engine: {self.db}")
        else:
            logger.error("Database is None on adapter!")

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

        logger.info(f"Functional units: {len(fu_list)} defined")
        # Always use compute_lcia_with_uncertainty, with n_simulations=1 for basic calculation
        return self.engine.compute_lcia_with_uncertainty(fu_list, n_simulations=n_simulations if uncertainty else 1)

def main(json_file_path: str):
    """Main function to run unified LCA calculation"""
    unified_engine = UnifiedLCAEngine()
    return unified_engine.run_lca_calculation(json_file_path)

if __name__ == "__main__":
    main('test.json')
