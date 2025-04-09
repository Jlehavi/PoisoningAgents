#!/usr/bin/env python
"""
Script to merge labeled data from multiple labelers.
This should be run after all labelers have completed their labeling task.
"""

import os
import argparse
from data_capture_and_label import IntersectionDataCapture

def main():
    """Main function to merge labeled data from multiple labelers."""
    parser = argparse.ArgumentParser(description='Merge labeled data from multiple labelers.')
    parser.add_argument('--labelers', nargs='+', default=["Sean", "Simon", "Jack"],
                       help='Names of labelers to merge data from')
    parser.add_argument('--output', type=str, default="merged_labels.json",
                       help='Output filename for merged labels')
    args = parser.parse_args()
    
    # Create data capture instance
    data_capture = IntersectionDataCapture()
    
    # Merge the labeled data
    print(f"Merging labeled data from: {', '.join(args.labelers)}")
    all_labels = data_capture.collect_and_merge_labels(args.labelers)
    
    if all_labels:
        print(f"Successfully merged {len(all_labels)} labels")
        
        # Print a summary of the labels
        print("\nLabel Summary:")
        
        # Count by labeler
        labeler_counts = {}
        for label in all_labels:
            labeler = label.get("labeler", "Unknown")
            labeler_counts[labeler] = labeler_counts.get(labeler, 0) + 1
        
        for labeler, count in labeler_counts.items():
            print(f"- {labeler}: {count} labels")
        
        # Count by action
        action_counts = {}
        for label in all_labels:
            action = label.get("best_move", "Unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print("\nLabels by action:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {action}: {count} labels")
        
        # Print example RAG usage
        print("\nMerged labels can now be used for RAG queries. Example:")
        print("  python data_capture_and_label.py --rag-query")
    else:
        print("No labels were merged. Please check if the labelers have completed their tasks.")

if __name__ == "__main__":
    main() 