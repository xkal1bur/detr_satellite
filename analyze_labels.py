import os
from collections import Counter
import json

def analyze_labels(annot_dir):
    # Dictionary to store label counts and mapping
    label_info = {
        'label_to_id': {},  # Map label names to numeric IDs
        'id_to_label': {},  # Map numeric IDs back to label names
        'counts': {},       # Count occurrences of each label
    }
    
    counter = Counter()
    
    # Process all annotation files
    for filename in sorted(os.listdir(annot_dir)):
        if not filename.endswith('.txt'):
            continue
            
        with open(os.path.join(annot_dir, filename), 'r') as f:
            # Skip first two lines (header and GSD)
            next(f)
            next(f)
            
            # Process each annotation line
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:  # Ensure we have enough parts
                    label = parts[8]
                    counter[label] += 1

    # Create label mappings (starting from 1, 0 reserved for background)
    for idx, (label, count) in enumerate(counter.most_common(), start=1):
        label_info['label_to_id'][label] = idx
        label_info['id_to_label'][str(idx)] = label  # Convert idx to str for JSON compatibility
        label_info['counts'][label] = count

    # Add background class
    label_info['label_to_id']['background'] = 0
    label_info['id_to_label']['0'] = 'background'
    label_info['counts']['background'] = 0

    # Save statistics
    print("\nLabel Statistics:")
    print(f"Total unique labels: {len(counter)}")
    print("\nTop 10 most common labels:")
    for label, count in counter.most_common(10):
        print(f"{label}: {count}")

    # Save to JSON file
    with open('label_info.json', 'w') as f:
        json.dump(label_info, f, indent=2, ensure_ascii=False)

    print(f"\nTotal number of classes (including background): {len(label_info['label_to_id'])}")
    print("Label information saved to label_info.json")

    return label_info

if __name__ == "__main__":
    data_dir = "data"
    train_labs_dir = os.path.join(data_dir, "train_labs")
    label_info = analyze_labels(train_labs_dir) 