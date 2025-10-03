# Dependencies: pip install pydantic pillow numpy opencv-python

import json
import re
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from collections import defaultdict

class ClassItem(BaseModel):
    """Pydantic model for a parsed class schedule item."""
    course_code: Optional[str] = None
    enrollment: Optional[str] = None         # "enrolled" | "waitlisted" | "planned" or None
    class_type: Optional[str] = None         # "lecture" | "lab" | "discussion" | None
    section: Optional[str] = None
    professor: Optional[str] = None
    room: Optional[str] = None
    building: Optional[str] = None
    time: Optional[str] = None               # normalized, e.g., "11:00-12:20" or "Tue 14:00-14:50"
    term: Optional[str] = None
    raw_text: str
    confidence: float                        # 0.0 - 1.0
    bbox: Optional[List[int]] = None         # [x0,y0,x1,y1]

def load_ocr_json(path: str) -> List[dict]:
    """Load OCR output JSON and return the blocks list."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('blocks', [])
    except FileNotFoundError:
        print(f"Error: {path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {path}")
        return []

def filter_ui_blocks(blocks: List[dict]) -> List[dict]:
    """Remove UI noise and low-confidence blocks."""
    ui_noise = {
        'remove', 'drop', 'change', 'calendar', 'print schedule', 'print',
        'schedule', 'export', 'view', 'edit', 'add', 'delete', 'save',
        'cancel', 'ok', 'yes', 'no', 'submit', 'close', 'menu', 'home',
        'back', 'next', 'previous', 'search', 'filter', 'sort'
    }
    
    filtered = []
    header_y_threshold = 100  # Assume header is in top 100 pixels
    
    for block in blocks:
        text = block.get('text', '').strip().lower()
        conf = block.get('conf', 0)
        top = block.get('top', 0)
        
        # Skip UI noise
        if text in ui_noise:
            continue
            
        # Skip very low confidence blocks
        if conf < 30:
            continue
            
        # Skip header area (adjust threshold as needed)
        if top < header_y_threshold and len(text) < 3:
            continue
            
        # Skip very short single characters unless they're numbers or letters
        if len(text) == 1 and not text.isalnum():
            continue
            
        filtered.append(block)
    
    return filtered

def group_blocks_into_tiles(blocks: List[dict]) -> List[dict]:
    """
    Spatially cluster blocks into tiles representing calendar boxes.
    Uses column detection followed by vertical grouping.
    """
    if not blocks:
        return []
    
    # Extract x-centroids for column detection
    centroids = []
    for block in blocks:
        left = block.get('left', 0)
        width = block.get('width', 0)
        centroid = left + width / 2
        centroids.append((centroid, block))
    
    # Simple column detection using histogram binning
    x_coords = [c[0] for c in centroids]
    if not x_coords:
        return []
        
    # Create histogram bins to detect columns
    min_x, max_x = min(x_coords), max(x_coords)
    num_bins = min(20, len(blocks))  # Adaptive bin count
    bin_width = (max_x - min_x) / num_bins if max_x > min_x else 1
    
    # Group blocks by approximate x-column
    columns = defaultdict(list)
    for centroid, block in centroids:
        bin_idx = int((centroid - min_x) / bin_width) if bin_width > 0 else 0
        columns[bin_idx].append(block)
    
    # Within each column, group vertically by overlapping y-ranges
    groups = []
    for col_blocks in columns.values():
        if not col_blocks:
            continue
            
        # Sort by y-position
        col_blocks.sort(key=lambda b: b.get('top', 0))
        
        # Group vertically overlapping blocks
        current_group = []
        current_y_range = None
        
        for block in col_blocks:
            top = block.get('top', 0)
            height = block.get('height', 0)
            bottom = top + height
            
            if current_y_range is None:
                # Start new group
                current_group = [block]
                current_y_range = (top, bottom)
            else:
                # Check if this block overlaps with current group's y-range
                group_top, group_bottom = current_y_range
                overlap_threshold = 10  # pixels
                
                if (top <= group_bottom + overlap_threshold and 
                    bottom >= group_top - overlap_threshold):
                    # Add to current group and expand y-range
                    current_group.append(block)
                    current_y_range = (min(group_top, top), max(group_bottom, bottom))
                else:
                    # Finalize current group and start new one
                    if current_group:
                        groups.append(create_group_from_blocks(current_group))
                    current_group = [block]
                    current_y_range = (top, bottom)
        
        # Don't forget the last group
        if current_group:
            groups.append(create_group_from_blocks(current_group))
    
    return groups

def create_group_from_blocks(blocks: List[dict]) -> dict:
    """Create a group dict from a list of blocks with combined text and bbox."""
    if not blocks:
        return {}
    
    # Sort blocks by top then left for reading order
    blocks.sort(key=lambda b: (b.get('top', 0), b.get('left', 0)))
    
    # Combine text
    texts = [block.get('text', '').strip() for block in blocks if block.get('text', '').strip()]
    combined_text = ' '.join(texts)
    
    # Calculate aggregated bbox
    lefts = [b.get('left', 0) for b in blocks]
    tops = [b.get('top', 0) for b in blocks]
    rights = [b.get('left', 0) + b.get('width', 0) for b in blocks]
    bottoms = [b.get('top', 0) + b.get('height', 0) for b in blocks]
    
    bbox = [min(lefts), min(tops), max(rights), max(bottoms)]
    
    # Average confidence
    confidences = [b.get('conf', 0) for b in blocks if b.get('conf', 0) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'text': combined_text,
        'bbox': bbox,
        'conf': avg_conf,
        'block_count': len(blocks)
    }

def merge_adjacent_tokens(line_text: str) -> str:
    """Helper to join split tokens like 'CSE' + '158R' into 'CSE158R'."""
    # Simple approach: remove spaces between uppercase letters and numbers/letters
    # Pattern: Letter(s) SPACE Number/Letter sequence
    merged = re.sub(r'([A-Z]+)\s+(\d+[A-Z]?)', r'\1\2', line_text)
    merged = re.sub(r'([A-Z]{2,4})\s+(\d{2,4})', r'\1\2', merged)
    return merged

def apply_ocr_corrections(text: str) -> str:
    """Apply common OCR error corrections."""
    corrections = {
        'ROS': 'R05',
        'R0S': 'R05',
        'ROB': 'R06',
        'R0B': 'R06',
        'R0G': 'R06',
    }
    
    # Apply targeted corrections
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Correct O->0 in room number patterns
    text = re.sub(r'\bR([O])(\d)', r'R0\2', text)
    
    # Correct S->5 in room numbers
    text = re.sub(r'\bR(\d)S\b', r'R\g<1>5', text)
    
    return text

def parse_group_to_class(group: dict) -> ClassItem:
    """Parse a grouped text block into a ClassItem using regex and heuristics."""
    raw_text = group.get('text', '').strip()
    bbox = group.get('bbox')
    confidence = group.get('conf', 0) / 100.0  # Convert to 0-1 scale
    
    # Apply OCR corrections
    text = apply_ocr_corrections(raw_text)
    text = merge_adjacent_tokens(text)
    
    # Clean up text
    text = ' '.join(text.split())  # Normalize whitespace
    text = re.sub(r'[^\w\s:/-]', ' ', text)  # Remove special chars except useful ones
    
    # Initialize fields
    course_code = None
    enrollment = None
    class_type = None
    section = None
    professor = None
    room = None
    building = None
    time = None
    term = None
    
    # Parse course code: ([A-Z]{2,4})\s*-?\s*(\d{2,4}[A-Z]?)
    course_match = re.search(r'\b([A-Z]{2,4})\s*-?\s*(\d{2,4}[A-Z]?)\b', text)
    if course_match:
        course_code = course_match.group(1) + course_match.group(2)
    
    # Parse time: find time ranges like "11:00-12:20" or single times
    time_patterns = [
        r'\b(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\b',  # Range
        r'\b(\d{1,2}:\d{2})\b'  # Single time
    ]
    
    for pattern in time_patterns:
        time_match = re.search(pattern, text)
        if time_match:
            if len(time_match.groups()) == 2:
                time = f"{time_match.group(1)}-{time_match.group(2)}"
            else:
                time = time_match.group(1)
            break
    
    # Parse enrollment status
    enrollment_keywords = {
        'enrolled': ['enrolled', 'enrolling'],
        'waitlisted': ['waitlist', 'waitlisted', 'wl'],
        'planned': ['planned', 'planning']
    }
    
    text_lower = text.lower()
    for status, keywords in enrollment_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            enrollment = status
            break
    
    # Parse class type and room/building
    # Look for patterns like "LE / RCLAS R05" or "DI / CSB 001"
    type_room_match = re.search(r'\b(LE|DI|LA|SE|ST|LEC|LAB|DIS)\s*/?\s*([A-Z]+)\s+([A-Z]?\d+[A-Z]?)\b', text)
    if type_room_match:
        class_type_abbrev = type_room_match.group(1)
        building = type_room_match.group(2)
        room = type_room_match.group(3)
        
        # Normalize class type
        type_mapping = {
            'LE': 'lecture', 'LEC': 'lecture',
            'DI': 'discussion', 'DIS': 'discussion',
            'LA': 'lab', 'LAB': 'lab',
            'SE': 'seminar', 'ST': 'studio'
        }
        class_type = type_mapping.get(class_type_abbrev, class_type_abbrev.lower())
    
    # Parse professor name (look for "Lastname, Firstname" pattern)
    prof_match = re.search(r'\b([A-Z][a-z]+),\s*([A-Z][a-z]+)\b', text)
    if prof_match:
        professor = f"{prof_match.group(1)}, {prof_match.group(2)}"
    
    # Parse section (look for SEC followed by letters/numbers)
    section_match = re.search(r'\bSEC?\s*([A-Z]?\d+[A-Z]?)\b', text)
    if section_match:
        section = section_match.group(1)
    
    # Parse term (look for terms like "Fall 2024", "Spring 2025")
    term_match = re.search(r'\b(Fall|Spring|Summer|Winter)\s+(\d{4})\b', text)
    if term_match:
        term = f"{term_match.group(1)} {term_match.group(2)}"
    
    return ClassItem(
        course_code=course_code,
        enrollment=enrollment,
        class_type=class_type,
        section=section,
        professor=professor,
        room=room,
        building=building,
        time=time,
        term=term,
        raw_text=raw_text,
        confidence=confidence,
        bbox=bbox
    )

def main():
    """Main processing pipeline."""
    print("Loading OCR data...")
    blocks = load_ocr_json('./ocr_output.json')
    print(f"Loaded {len(blocks)} blocks from OCR")
    
    print("Filtering UI blocks...")
    filtered_blocks = filter_ui_blocks(blocks)
    print(f"Filtered to {len(filtered_blocks)} content blocks")
    
    print("Grouping blocks into tiles...")
    groups = group_blocks_into_tiles(filtered_blocks)
    print(f"Created {len(groups)} spatial groups")
    
    print("Parsing groups into classes...")
    parsed_classes = []
    ambiguous_groups = []
    
    for group in groups:
        try:
            class_item = parse_group_to_class(group)
            
            # Check if this should be considered ambiguous
            is_ambiguous = (
                class_item.confidence < 0.7 or
                (not class_item.course_code and not class_item.time) or
                len(class_item.raw_text.strip()) < 3
            )
            
            if is_ambiguous:
                ambiguous_groups.append({
                    'raw_text': group.get('text', ''),
                    'bbox': group.get('bbox'),
                    'confidence': group.get('conf', 0) / 100.0,
                    'reason': 'low_confidence_or_missing_fields'
                })
            else:
                parsed_classes.append(class_item.dict())
                
        except Exception as e:
            print(f"Error parsing group: {e}")
            ambiguous_groups.append({
                'raw_text': group.get('text', ''),
                'bbox': group.get('bbox'),
                'confidence': group.get('conf', 0) / 100.0,
                'reason': f'parsing_error: {str(e)}'
            })
    
    # Write output files
    print("Writing output files...")
    with open('parsed_classes.json', 'w', encoding='utf-8') as f:
        json.dump(parsed_classes, f, indent=2, ensure_ascii=False)
    
    with open('ambiguous_groups.json', 'w', encoding='utf-8') as f:
        json.dump(ambiguous_groups, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== PARSING SUMMARY ===")
    print(f"Blocks loaded: {len(blocks)}")
    print(f"Groups created: {len(groups)}")
    print(f"Classes parsed: {len(parsed_classes)}")
    print(f"Ambiguous groups: {len(ambiguous_groups)}")
    print(f"Output written to: parsed_classes.json, ambiguous_groups.json")

if __name__ == "__main__":
    main()
