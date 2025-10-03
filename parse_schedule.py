#!/usr/bin/env python3
"""
University Schedule Parser using OpenAI GPT-4 Vision
Analyzes schedule images and outputs structured JSON with class information.
"""

import os
import json
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_schedule(image_path):
    """
    Parse a university schedule image using OpenAI GPT-4 Vision.

    Args:
        image_path: Path to the schedule image

    Returns:
        dict: Structured JSON with class information
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Encode the image
    base64_image = encode_image(image_path)

    # Create the prompt for GPT-4 Vision
    prompt = """Analyze this university schedule image and extract all class information into a structured JSON format.

CRITICAL: This is a weekly calendar grid view. Days of the week are arranged in COLUMNS from left to right:
- Column 1 (leftmost) = Monday
- Column 2 = Tuesday
- Column 3 = Wednesday
- Column 4 = Thursday
- Column 5 = Friday
- Column 6 = Saturday (if present)
- Column 7 (rightmost) = Sunday (if present)

Day labels may appear at the top of each column. If not visible, determine the day based on which column the class appears in.

For each class tile/block you see in the schedule, extract:
- course_code: Course code (e.g., "CSE158R") in uppercase with no spaces, or null if not visible
- enrollment: Status as "enrolled", "waitlisted", "planned", or null
- class_type: Type as "lecture", "lab", "discussion", or null (LE=lecture, DI=discussion, LA=lab)
- section: Section code (e.g., "R05") or null
- professor: Professor name exactly as written, or null if not shown
- room: Room code exactly as shown (e.g., "RCLAS R05", "WFH 1N108") or null
- building: Building code if identifiable (e.g., "RCLAS", "WFH") or null
- day_of_week: List of days ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] or empty list if not visible
- start_time: Start time in strict 24-hour "HH:MM" format (e.g., "13:00", "09:30") or null - MUST be pandas-compatible
- end_time: End time in strict 24-hour "HH:MM" format (e.g., "14:20", "10:50") or null - MUST be pandas-compatible
- term: Term identifier if visible (e.g., "2025-FA") or null
- raw_text: Exact concatenated text from the tile
- confidence: Your confidence level from 0.0 to 1.0
- bbox: Pixel coordinates [x0, y0, x1, y1] as integers for the class tile location, or null

Return ONLY valid JSON in this exact format:
{
  "classes": [
    {
      "course_code": "string or null",
      "enrollment": "enrolled|waitlisted|planned|null",
      "class_type": "lecture|lab|discussion|null",
      "section": "string or null",
      "professor": "string or null",
      "room": "string or null",
      "building": "string or null",
      "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri"] or [],
      "start_time": "HH:MM or null",
      "end_time": "HH:MM or null",
      "term": "string or null",
      "raw_text": "string",
      "confidence": 0.0-1.0,
      "bbox": [x0, y0, x1, y1] or null
    }
  ],
  "metadata": {
    "source_image_id": "filename or null",
    "parser_version": "claude-image-v1"
  }
}

CRITICAL DAY/TIME FORMATTING REQUIREMENTS:
- day_of_week is THE MOST IMPORTANT field - you MUST get this correct!
- day_of_week MUST be determined by which COLUMN the class appears in (left to right: Mon, Tue, Wed, Thu, Fri, Sat, Sun)
- If a class spans multiple columns, include all relevant days (e.g., a class in columns 2 and 4 = ["Tue", "Thu"])
- day_of_week MUST be a list containing relevant day abbreviations from ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
- DO NOT guess days - look at the column position or visible day labels
- start_time and end_time MUST be in strict 24-hour HH:MM format (e.g., "13:00", "09:30")
- Convert ALL times to 24-hour format: 1:00 PM → "13:00", 9:00 AM → "09:00"
- NEVER use 12-hour format (no AM/PM in the time fields)
- Ensure times are zero-padded (e.g., "09:00" not "9:00", "09:30" not "9:30")
- These formats are required for pandas time series processing

Important:
- Extract ALL class blocks you see in the calendar
- Be precise with course codes, removing spaces (e.g., "CSE 158R" → "CSE158R")
- Identify building codes from room strings (e.g., "RCLAS R05" → building: "RCLAS", room: "RCLAS R05")
- Set confidence based on text clarity (clear=0.9-1.0, slightly unclear=0.7-0.9, unclear=0.5-0.7)
- Return ONLY the JSON, no additional text or explanations"""

    # Make API call to GPT-4 Vision
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=4096
    )

    # Extract and parse the response
    result_text = response.choices[0].message.content.strip()

    # Debug: print raw response
    print(f"DEBUG - Raw response: {result_text[:500]}")

    # Remove markdown code blocks if present
    if result_text.startswith("```"):
        result_text = result_text.split("```")[1]
        if result_text.startswith("json"):
            result_text = result_text[4:]
        result_text = result_text.strip()

    # Parse JSON
    result = json.loads(result_text)

    # Update metadata with actual filename
    result["metadata"]["source_image_id"] = Path(image_path).name

    return result

def main():
    """Main function to run the schedule parser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_schedule.py <image_path> [output_json_path]")
        print("\nExample: python parse_schedule.py test_sched.png output.json")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)

    print(f"Parsing schedule from: {image_path}")

    try:
        result = parse_schedule(image_path)

        # Pretty print the result
        formatted_json = json.dumps(result, indent=2)
        print("\nParsed Schedule:")
        print(formatted_json)

        # Save to file if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                f.write(formatted_json)
            print(f"\nSaved to: {output_path}")

    except Exception as e:
        print(f"Error parsing schedule: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
