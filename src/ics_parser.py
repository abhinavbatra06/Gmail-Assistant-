"""
ICS (iCalendar) Parser Module

Parses .ics files and extracts structured event data.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional
from email.utils import parsedate_to_datetime


class ICSParser:
    
    @staticmethod
    def parse_ics_file(ics_path: str) -> List[Dict]:
        """
        Parse an ICS file and extract all events.
        
        Args:
            ics_path: Path to .ics file
            
        Returns:
            List of event dictionaries with structured data
        """
        events = []
        
        try:
            with open(ics_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # split into individual VEVENT blocks
            vevent_blocks = re.split(r'BEGIN:VEVENT', content)
            
            for block in vevent_blocks[1:]:  # skip first empty split
                event = ICSParser._parse_vevent(block)
                if event:
                    events.append(event)
        
        except Exception as e:
            print(f"Error parsing ICS file {ics_path}: {str(e)}")
        
        return events
    
    @staticmethod
    def _parse_vevent(vevent_block: str) -> Optional[Dict]:
        """
        Parse a single VEVENT block.
        
        Args:
            vevent_block: Text content of a VEVENT block
            
        Returns:
            Event dictionary or None if parsing fails
        """
        event = {}
        
        # extract key-value pairs
        lines = vevent_block.split('\n')
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line or line == 'END:VEVENT':
                continue
            
            # handle line continuation (starts with space)
            if line.startswith(' '):
                if current_key:
                    current_value.append(line[1:])
                continue
            
            # save previous key-value if exists
            if current_key:
                value = ''.join(current_value)
                event[current_key] = ICSParser._clean_ical_value(value)
                current_value = []
            
            # parse new key-value pair
            if ':' in line:
                key, value = line.split(':', 1)
                # handle parameters (e.g., DTSTART;TZID=America/New_York:20231201T120000)
                if ';' in key:
                    key = key.split(';')[0]
                
                current_key = key
                current_value = [value]
        
        # save last key-value
        if current_key:
            value = ''.join(current_value)
            event[current_key] = ICSParser._clean_ical_value(value)
        
        # return only if we have essential fields
        if 'SUMMARY' in event or 'DTSTART' in event:
            return ICSParser._structure_event(event)
        
        return None
    
    @staticmethod
    def _clean_ical_value(value: str) -> str:
        """
        Clean iCalendar value (remove escape sequences).
        
        Args:
            value: Raw iCalendar value
            
        Returns:
            Cleaned value
        """
        # remove common escape sequences
        value = value.replace('\\n', '\n')
        value = value.replace('\\,', ',')
        value = value.replace('\\;', ';')
        value = value.replace('\\\\', '\\')
        return value.strip()
    
    @staticmethod
    def _parse_datetime(dt_string: str) -> Optional[datetime]:
        """
        Parse iCalendar datetime string.
        
        Args:
            dt_string: iCalendar datetime string (e.g., "20231201T120000Z" or "20231201T120000")
            
        Returns:
            datetime object or None if parsing fails
        """
        if not dt_string:
            return None
        
        try:
            # remove timezone indicator if present
            dt_string = dt_string.replace('Z', '')
            
            # handle format: YYYYMMDDTHHMMSS
            if 'T' in dt_string and len(dt_string) >= 15:
                date_part = dt_string.split('T')[0]
                time_part = dt_string.split('T')[1]
                
                if len(date_part) == 8 and len(time_part) >= 6:
                    year = int(date_part[0:4])
                    month = int(date_part[4:6])
                    day = int(date_part[6:8])
                    hour = int(time_part[0:2])
                    minute = int(time_part[2:4])
                    second = int(time_part[4:6]) if len(time_part) >= 6 else 0
                    
                    return datetime(year, month, day, hour, minute, second)
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def _structure_event(event: Dict) -> Dict:
        """
        Structure event data into a consistent format.
        
        Args:
            event: Raw event dictionary from parsing
            
        Returns:
            Structured event dictionary
        """
        structured = {
            "summary": event.get("SUMMARY", "Untitled Event"),
            "description": event.get("DESCRIPTION", ""),
            "location": event.get("LOCATION", ""),
            "organizer": event.get("ORGANIZER", "").replace("mailto:", ""),
            "start_time": None,
            "end_time": None,
            "start_time_str": event.get("DTSTART", ""),
            "end_time_str": event.get("DTEND", ""),
            "uid": event.get("UID", ""),
            "status": event.get("STATUS", "CONFIRMED"),
            "raw_event": event  # keep raw data for reference
        }
        
        # parse datetimes
        if structured["start_time_str"]:
            structured["start_time"] = ICSParser._parse_datetime(structured["start_time_str"])
        
        if structured["end_time_str"]:
            structured["end_time"] = ICSParser._parse_datetime(structured["end_time_str"])
        
        # create searchable text representation
        text_parts = [
            f"Event: {structured['summary']}",
            f"Location: {structured['location']}" if structured['location'] else "",
            f"Description: {structured['description']}" if structured['description'] else "",
            f"Start: {structured['start_time_str']}",
            f"End: {structured['end_time_str']}" if structured['end_time_str'] else "",
            f"Organizer: {structured['organizer']}" if structured['organizer'] else ""
        ]
        structured["text"] = "\n".join([p for p in text_parts if p])
        
        return structured

