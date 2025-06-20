import json

COLUMN_NAMES = ["file_name", "processed"]
TRACKER_FILE = "../modules/tracker.json"

def add_new_file(filename: str) -> None:
    data = get_tracker_data()
    existing = [item['file_name'] for item in data]
    if filename not in existing:
        data.append({"file_name": filename})
        with open(TRACKER_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        print(f"File '{filename}' already exists in the tracker.")
        
def mark_file_processed(filename: str, field_name: str) -> None:
    data = get_tracker_data()
    
    with open(TRACKER_FILE, 'w') as f:
        for item in data:
            if item['file_name'] == filename:
                item[field_name] = True
                break
        else:
            print(f"File '{filename}' not found in the tracker.")
            return
        
        json.dump(data, f, indent=4)
        
def file_exists(filename: str) -> bool:
    data = get_tracker_data()
    return any(item['file_name'] == filename for item in data)
        
def is_file_processed(filename: str, field_name: str) -> bool:
    data = get_tracker_data()
    for item in data:
        if item['file_name'] == filename:
            return item.get(field_name, False)
    return False
    
def get_tracker_data() -> list:
    with open(TRACKER_FILE, 'r') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Tracker data is not a list.")
            return data
        except json.JSONDecodeError:
            print("Tracker file is empty or corrupted.")
            return []
        except ValueError as e:
            print(f"Error reading tracker data: {e}")
            return []
    
# Testing
if __name__ == "__main__":
    add_new_file("example.txt")
    mark_file_processed("example.txt", "bert_embedding")
    add_new_file("example.txt")
    print(is_file_processed("example.txt", "bert_embedding"))
    print(is_file_processed("nonexistent.txt", "bert_embedding"))
    
    # Add a breakpoint in between to inspect the tracker file
    
    with open(TRACKER_FILE, 'w') as f:
        f.write("")