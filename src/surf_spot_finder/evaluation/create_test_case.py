#!/usr/bin/env python3
import os
import yaml
from datetime import datetime, timedelta
import argparse
from pathlib import Path

def get_input(prompt, default=None, required=False):
    """Get input from user with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        value = input(prompt)
        if not value and default:
            return default
        elif not value and required:
            print("This field is required. Please enter a value.")
        else:
            return value or None

def create_test_case():
    print("\n===== Surf Spot Finder Test Case Creator =====\n")
    
    # Test case file name
    test_name = get_input("Enter test case name (will be used as filename)", required=True)
    filename = f"{test_name.lower().replace(' ', '_')}.yaml"
    
    # Create test case structure
    test_case = {
        "input": {},
        "ground_truth": [],
        "checkpoints": []
    }
    
    # Gather input data
    print("\n=== Input Data ===")
    test_case["input"]["location"] = get_input("Enter location (e.g., 'Vigo')", required=True)
    
    # Date suggestion (3 days from now)
    suggested_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d %H:%M")
    test_case["input"]["date"] = get_input("Enter date (format: YYYY-MM-DD HH:MM)", suggested_date)
    
    test_case["input"]["max_driving_hours"] = get_input("Enter maximum driving hours", "3")
    test_case["input"]["json_tracer"] = True
    
    # Gather ground truth
    print("\n=== Ground Truth ===")
    print("Add ground truth items (press Enter with empty name to finish):")
    
    while True:
        name = get_input("Name (e.g., 'Surf location')")
        if not name:
            break
        
        points = get_input("Points (integer value)", "5")
        try:
            points = int(points)
        except ValueError:
            points = 5
            print("Invalid points value, using default: 5")
        
        value = get_input("Expected value", required=True)
        
        test_case["ground_truth"] = {
            "name": name,
            "points": points,
            "value": value
        }
    
    # Gather checkpoints
    print("\n=== Checkpoints ===")
    print("Add checkpoint criteria (press Enter with empty criteria to finish):")
    
    checkpoint_examples = [
        "Check if the agent did a web search for nearby surf locations.",
        "Check if the agent used the get_surfing_spots tool and it succeeded",
        "Check if the agent used the get_wave_forecast tool and it succeeded",
        "Check if the agent used the get_wind_forecast tool and it succeeded",
        "Check if the agent used the get_area_lat_lon tool and it succeeded",
        "Check if the final answer contains any description about the weather at the chosen location"
    ]
    
    print("\nExample checkpoints:")
    for i, example in enumerate(checkpoint_examples, 1):
        print(f"{i}. {example}")
    
    while True:
        criteria = get_input("Checkpoint criteria (or enter a number 1-6 to use an example)")
        
        if not criteria:
            break
        
        # Check if user entered a number to use an example
        try:
            example_num = int(criteria)
            if 1 <= example_num <= len(checkpoint_examples):
                criteria = checkpoint_examples[example_num - 1]
        except ValueError:
            pass
        
        points = get_input("Points for this checkpoint", "1")
        try:
            points = int(points)
        except ValueError:
            points = 1
            print("Invalid points value, using default: 1")
        
        test_case["checkpoints"].append({
            "points": points,
            "criteria": criteria
        })
    
    # Save the test case to a file
    output_dir = Path(__file__).parent / "test_cases"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        f.write(f"# Test case for surf spot finder\n\n")
        f.write("# You only need this input data if you want to run the test case, if you pass in a path to a telemetry file this\n")
        f.write("# is ignored\n")
        yaml.dump(test_case, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nTest case created successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a YAML test case for the Surf Spot Finder")
    args = parser.parse_args()
    create_test_case()