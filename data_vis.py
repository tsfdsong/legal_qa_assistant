import json
import sys

def decode_unicode(obj):
    """递归解码字典中的Unicode字符串"""
    if isinstance(obj, dict):
        return {decode_unicode(key): decode_unicode(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [decode_unicode(item) for item in obj]
    elif isinstance(obj, str):
        # 尝试解码Unicode转义序列
        try:
            return obj.encode('latin1').decode('unicode_escape')
        except:
            return obj
    return obj

def main(json_file):
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解码整个数据结构
        decoded_data = decode_unicode(data)
        
        # 检查是否有docstore/data部分
        if "docstore/data" not in decoded_data:
            print("错误：JSON文件中未找到'docstore/data'部分")
            return
        
        # 提取并打印中文内容
        print("=" * 80)
        print(f"成功解析文件: {json_file}")
        print(f"找到 {len(decoded_data['docstore/data'])} 个条目")
        print("=" * 80)
        
        for key, node in decoded_data["docstore/data"].items():
            # 确保节点结构正确
            if "__data__" in node and isinstance(node["__data__"], dict):
                node_data = node["__data__"]
                print(f"\n节点ID: {node_data.get('id_', 'N/A')}")
                
                # 打印元数据
                metadata = node_data.get("metadata", {})
                print(f"法律名称: {metadata.get('law_name', 'N/A')}")
                print(f"条款: {metadata.get('article', 'N/A')}")
                print(f"来源文件: {metadata.get('source_file', 'N/A')}")
                
                # 打印内容
                content = node_data.get("text", "")
                print(f"内容: {content}")
                
                print("-" * 60)
            else:
                print(f"\n警告: 跳过无效节点 - {key}")
        
        print("\n解析完成!")
    
    except FileNotFoundError:
        print(f"错误: 文件 '{json_file}' 未找到")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{json_file}' 不是有效的JSON格式")
    except Exception as e:
        print(f"发生未知错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python decode_json.py <json文件路径>")
        print("示例: python decode_json.py data.json")
    else:
        main(sys.argv[1])