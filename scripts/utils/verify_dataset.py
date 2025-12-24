import os
from pathlib import Path

def file_exists_and_not_empty(filepath):
    return os.path.isfile(filepath) and os.path.getsize(filepath) > 0

def check_example_complement(example_path):
    input_path = os.path.join(example_path, "input")
    output_path = os.path.join(example_path, "output")
    config_path = os.path.join(example_path, "env_config")
    
    if not os.path.exists(input_path) or not os.path.exists(output_path) or not os.path.exists(config_path):
        return False
        
    files_to_check = [
        os.path.join(input_path, "input.png"),
        os.path.join(input_path, "input_mask.png"),
        os.path.join(input_path, "instruction.txt"),
        os.path.join(output_path, "operation_sequence.json"),
        os.path.join(config_path, "env_config.json")
    ]
    
    for f in files_to_check:
        if not file_exists_and_not_empty(f):
            return False
    return True

def main():
    # 获取脚本所在目录的上级目录作为仓库根目录
    script_root = Path(__file__).resolve().parents[2]
    data_path = str(script_root / "dataset" / "vlm_evaluation_v1.0")
    error_example_list = {}
    number_of_examples = 0
    
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist.")
        return

    # 预期的维度列表
    expected_dims = ["M&T", "CommenSence", "Complex", "PhysicsLaw", "Semantic", "Spatial"]
    
    found_dims = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')]
    print(f"Found dimensions: {found_dims}")
    
    for dim in found_dims:
        dim_path = os.path.join(data_path, dim)
        for task in os.listdir(dim_path):
            task_path = os.path.join(dim_path, task)
            if not os.path.isdir(task_path):
                continue
            
            for example in os.listdir(task_path):
                example_path = os.path.join(task_path, example)
                if not os.path.isdir(example_path):
                    continue
                
                if not check_example_complement(example_path):
                    if task not in error_example_list:
                        error_example_list[task] = []
                    error_example_list[task].append(f"{dim}/{task}/{example}")
                else:
                    number_of_examples += 1

    if len(error_example_list) == 0:
        print("✅ All examples are complete!")
        print(f"Total number of valid examples: {number_of_examples}")
    else:
        print(f"❌ Number of Normal examples: {number_of_examples}")
        print(f"⚠️ Some examples are incomplete ({len(error_example_list)} tasks affected)!")
        for task, examples in error_example_list.items():
            print(f"Task: {task} - {len(examples)} incomplete examples")
            # 只打印前几个例子避免输出过多
            for ex in examples[:3]:
                print(f"  - {ex}")
            if len(examples) > 3:
                print(f"  - ... and {len(examples)-3} more")

if __name__ == "__main__":
    main()
