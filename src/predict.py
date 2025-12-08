"""
Predict Module - Structured Event Database

Maintains a structured database of calendar events extracted from ICS files and email body text.
Provides direct querying without vector search for calendar-related queries.
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from src.ics_parser import ICSParser
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import os as os_module
    load_dotenv()
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class Predict:
    
    def __init__(self, db_path: str = "db/events.db", openai_api_key: Optional[str] = None):
        """
        Initialize Predict module.
        
        Args:
            db_path: Path to SQLite database for events
            openai_api_key: Optional OpenAI API key for LLM-based event extraction
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
        # init openai client for event extraction from text
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = openai_api_key or os_module.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
    
    def _create_tables(self):
        """Create database tables for events."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uid TEXT UNIQUE,
                summary TEXT,
                description TEXT,
                location TEXT,
                organizer TEXT,
                start_time DATETIME,
                end_time DATETIME,
                start_time_str TEXT,
                end_time_str TEXT,
                status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # table to track which emails contain which events (many-to-many)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS event_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                message_id TEXT,
                attachment_filename TEXT,
                email_date TEXT,
                is_primary BOOLEAN DEFAULT 0,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                UNIQUE(event_id, message_id)
            )
        """)
        
        # index for fast date queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_start_time ON events(start_time)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_end_time ON events(end_time)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_emails_event ON event_emails(event_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_emails_message ON event_emails(message_id)
        """)
        # index for fuzzy matching
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_start ON events(summary, start_time)
        """)
        
        self.conn.commit()
    
    def _parse_datetime_from_db(self, dt_value) -> Optional[datetime]:
        """
        Parse datetime value from database (SQLite returns DATETIME as string).
        
        Args:
            dt_value: Datetime value from database (string or datetime object)
            
        Returns:
            datetime object or None if parsing fails
        """
        if dt_value is None:
            return None
        
        if isinstance(dt_value, datetime):
            return dt_value
        
        if isinstance(dt_value, str):
            # try ISO format first
            try:
                return datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # try SQLite datetime format: "YYYY-MM-DD HH:MM:SS"
                try:
                    return datetime.strptime(dt_value, "%Y-%m-%d %H:%M:%S")
                except:
                    # try date only format
                    try:
                        return datetime.strptime(dt_value, "%Y-%m-%d")
                    except:
                        return None
        
        return None
    
    def _convert_event_datetimes(self, event: Dict) -> Dict:
        """
        Convert datetime strings in event dict to datetime objects.
        
        Args:
            event: Event dictionary from database
            
        Returns:
            Event dictionary with datetime objects
        """
        if event.get("start_time"):
            event["start_time"] = self._parse_datetime_from_db(event["start_time"])
        
        if event.get("end_time"):
            event["end_time"] = self._parse_datetime_from_db(event["end_time"])
        
        return event
    
    def extract_events_from_email(self, message_id: str, attachment_path: str, 
                                  attachment_filename: str, email_date: Optional[str] = None) -> int:
        """
        Extract events from an ICS attachment and store in database with deduplication.
        
        Args:
            message_id: Email message ID
            attachment_path: Path to ICS file
            attachment_filename: Name of attachment file
            email_date: Date of the email (for determining primary source)
            
        Returns:
            Number of events extracted
        """
        events = ICSParser.parse_ics_file(attachment_path)
        count = 0
        
        for event in events:
            try:
                event_id = self._store_event_with_deduplication(
                    event, message_id, attachment_filename, email_date
                )
                if event_id:
                    count += 1
            except Exception as e:
                print(f"Error storing event {event.get('uid', 'unknown')}: {str(e)}")
        
        self.conn.commit()
        return count
    
    def extract_events_from_text(self, message_id: str, email_text: str, 
                                 email_date: Optional[str] = None, 
                                 email_subject: Optional[str] = None) -> int:
        """
        Extract events from email body text using LLM.
        This handles cases where events are mentioned in email text but not in ICS attachments.
        
        Args:
            message_id: Email message ID
            email_text: Email body text
            email_date: Date of the email
            email_subject: Subject of the email
            
        Returns:
            Number of events extracted
        """
        if not self.openai_client:
            return 0
        
        if not email_text or len(email_text.strip()) < 50:
            return 0
        
        try:
            # use LLM to extract event information from text
            prompt = f"""Extract calendar event information from the following text (which may be from an email body or attachment like an image/PDF).
Return a JSON object with an "events" array, where each event has:
- summary: Event title/name
- description: Event description (if available)
- location: Event location/venue (be specific, e.g., "NYU MakerSpace Design Lab", "Makerspace Design Lab")
- start_time_str: Start date/time as string (e.g., "2025-12-05 14:00" or "December 5, 2025 at 2:00 PM")
- end_time_str: End date/time as string (if available)
- organizer: Organizer name or email (if mentioned)

Important:
- Extract ALL events mentioned, even if details are incomplete
- For location, use the exact name mentioned (e.g., "MakerSpace Design Lab", "NYU MakerSpace", "Design Lab")
- If the text mentions "upcoming events" or "events at [location]", extract each event separately
- If dates are relative (e.g., "this week", "next week"), calculate the actual date based on the email date

If no events are found, return {{"events": []}}.

Email subject: {email_subject or 'N/A'}
Email date: {email_date or 'N/A'}

Text content:
{email_text[:4000]}  # limit text length

Return ONLY valid JSON object with "events" array, no other text:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a calendar event extraction assistant. Extract event information from email text and return as JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            events = result.get("events", [])
            
            if not isinstance(events, list):
                events = []
            
            count = 0
            for event_data in events:
                try:
                    # convert to standard event format
                    event = {
                        "uid": f"text_{message_id}_{count}",  # generate UID
                        "summary": event_data.get("summary", ""),
                        "description": event_data.get("description", ""),
                        "location": event_data.get("location", ""),
                        "organizer": event_data.get("organizer", ""),
                        "start_time_str": event_data.get("start_time_str", ""),
                        "end_time_str": event_data.get("end_time_str", ""),
                        "status": "CONFIRMED"
                    }
                    
                    # parse start_time if available
                    if event["start_time_str"]:
                        try:
                            from email.utils import parsedate_to_datetime
                            event["start_time"] = parsedate_to_datetime(event["start_time_str"])
                        except:
                            # try parsing as ISO format or common date formats
                            try:
                                event["start_time"] = datetime.fromisoformat(event["start_time_str"].replace("Z", "+00:00"))
                            except:
                                event["start_time"] = None
                    else:
                        event["start_time"] = None
                    
                    # parse end_time if available
                    if event["end_time_str"]:
                        try:
                            from email.utils import parsedate_to_datetime
                            event["end_time"] = parsedate_to_datetime(event["end_time_str"])
                        except:
                            try:
                                event["end_time"] = datetime.fromisoformat(event["end_time_str"].replace("Z", "+00:00"))
                            except:
                                event["end_time"] = None
                    else:
                        event["end_time"] = None
                    
                    # store event (use "attachment" as source if it is from attachment text)
                    source_type = "email_body" if "Attachment:" not in email_text[:100] else "attachment"
                    event_id = self._store_event_with_deduplication(
                        event, message_id, source_type, email_date
                    )
                    if event_id:
                        count += 1
                except Exception as e:
                    print(f"Error storing extracted event: {str(e)}")
                    continue
            
            self.conn.commit()
            return count
            
        except Exception as e:
            print(f"Error extracting events from text: {str(e)}")
            return 0
    
    def _store_event_with_deduplication(self, event: Dict, message_id: str, 
                                        attachment_filename: str, email_date: Optional[str] = None) -> Optional[int]:
        """
        Store event with intelligent deduplication.
        
        Deduplication strategy:
        1. Try to match by UID (exact match)
        2. If no UID or no match, try fuzzy matching (summary + start_time)
        3. If match found, link this email to existing event
        4. If no match, create new event
        
        Args:
            event: Event dictionary from parser
            message_id: Email message ID
            attachment_filename: Attachment filename
            email_date: Email date string
            
        Returns:
            Event ID (existing or new)
        """
        cur = self.conn.cursor()
        event_uid = event.get("uid", "")
        summary = event.get("summary", "")
        start_time = event.get("start_time")
        
        # try to find by UID
        event_id = None
        if event_uid:
            cur.execute("SELECT id FROM events WHERE event_uid = ?", (event_uid,))
            row = cur.fetchone()
            if row:
                event_id = row[0]
        
        # fuzzy matching if no UID match
        if not event_id and summary and start_time:
            # match by summary (normalized) and start_time (within 1 minute tolerance)
            normalized_summary = " ".join(summary.lower().split())
            
            # find events with same summary and similar start time
            cur.execute("""
                SELECT id, summary, start_time 
                FROM events 
                WHERE LOWER(TRIM(REPLACE(REPLACE(summary, '  ', ' '), '  ', ' '))) = ?
                AND start_time IS NOT NULL
                AND ABS(JULIANDAY(start_time) - JULIANDAY(?)) < 0.0007
            """, (normalized_summary, start_time))
            
            rows = cur.fetchall()
            if rows:
                # use the first match (could enhance to pick best match)
                event_id = rows[0][0]
        
        # if event exists, just link this email to it
        if event_id:
            # check if this email is already linked
            cur.execute("""
                SELECT id FROM event_emails 
                WHERE event_id = ? AND message_id = ?
            """, (event_id, message_id))
            
            if not cur.fetchone():
                # link email to event
                cur.execute("""
                    INSERT INTO event_emails 
                    (event_id, message_id, attachment_filename, email_date, is_primary)
                    VALUES (?, ?, ?, ?, 0)
                """, (event_id, message_id, attachment_filename, email_date))
                
                # update event if this email has more complete information
                self._update_event_if_newer(event_id, event, email_date)
        else:
            # create new event
            cur.execute("""
                INSERT INTO events 
                (event_uid, summary, description, location, organizer, 
                 start_time, end_time, start_time_str, end_time_str, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_uid,
                summary,
                event.get("description", ""),
                event.get("location", ""),
                event.get("organizer", ""),
                start_time,
                event.get("end_time"),
                event.get("start_time_str", ""),
                event.get("end_time_str", ""),
                event.get("status", "CONFIRMED")
            ))
            event_id = cur.lastrowid
            
            # link email to event (mark as primary if it's the first)
            cur.execute("""
                INSERT INTO event_emails 
                (event_id, message_id, attachment_filename, email_date, is_primary)
                VALUES (?, ?, ?, ?, 1)
            """, (event_id, message_id, attachment_filename, email_date))
        
        return event_id
    
    def _update_event_if_newer(self, event_id: int, new_event: Dict, email_date: Optional[str]):
        """
        Update event with information from newer email if it's more complete.
        
        Args:
            event_id: Existing event ID
            new_event: New event data
            email_date: Date of the new email
        """
        cur = self.conn.cursor()
        
        # get current event
        cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        columns = [desc[0] for desc in cur.description]
        current = dict(zip(columns, cur.fetchone()))
        
        # update fields that are empty in current but present in new
        updates = []
        params = []
        
        if not current.get("description") and new_event.get("description"):
            updates.append("description = ?")
            params.append(new_event["description"])
        
        if not current.get("location") and new_event.get("location"):
            updates.append("location = ?")
            params.append(new_event["location"])
        
        if not current.get("end_time") and new_event.get("end_time"):
            updates.append("end_time = ?")
            params.append(new_event["end_time"])
            updates.append("end_time_str = ?")
            params.append(new_event.get("end_time_str", ""))
        
        # update status if new one is more recent (e.g., CANCELLED overrides CONFIRMED)
        if new_event.get("status") in ["CANCELLED", "TENTATIVE"]:
            updates.append("status = ?")
            params.append(new_event["status"])
        
        if updates:
            params.append(event_id)
            cur.execute(f"""
                UPDATE events 
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, params)
    
    def query_events(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    location: Optional[str] = None,
                    organizer: Optional[str] = None,
                    summary_keywords: Optional[List[str]] = None,
                    limit: int = 50,
                    deduplicate: bool = True) -> List[Dict]:
        """
        Query events from the database with deduplication.
        
        Args:
            start_date: Filter events starting on or after this date
            end_date: Filter events ending on or before this date
            location: Filter by location (contains)
            organizer: Filter by organizer email
            summary_keywords: Filter by keywords in summary
            limit: Maximum number of results
            deduplicate: If True, return only unique events (default: True)
            
        Returns:
            List of event dictionaries (deduplicated)
        """
        cur = self.conn.cursor()
        query = """
            SELECT DISTINCT e.*, 
                   GROUP_CONCAT(DISTINCT ee.message_id) as message_ids,
                   COUNT(DISTINCT ee.message_id) as email_count
            FROM events e
            LEFT JOIN event_emails ee ON e.id = ee.event_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND (e.end_time IS NULL OR e.end_time >= ?)"
            params.append(start_date)
        
        if end_date:
            query += " AND (e.start_time IS NULL OR e.start_time <= ?)"
            params.append(end_date)
        
        if location:
            query += " AND e.location LIKE ?"
            params.append(f"%{location}%")
        
        if organizer:
            query += " AND e.organizer LIKE ?"
            params.append(f"%{organizer}%")
        
        if summary_keywords:
            for keyword in summary_keywords:
                query += " AND e.summary LIKE ?"
                params.append(f"%{keyword}%")
        
        query += " GROUP BY e.id ORDER BY e.start_time ASC LIMIT ?"
        params.append(limit)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        # get column names
        columns = [desc[0] for desc in cur.description]
        
        events = []
        for row in rows:
            event = dict(zip(columns, row))
            # parse message_ids string into list
            if event.get("message_ids"):
                event["message_ids"] = event["message_ids"].split(",")
            else:
                event["message_ids"] = []
            
            # convert datetime strings back to datetime objects for proper comparison
            # SQLite returns DATETIME columns as strings
            event = self._convert_event_datetimes(event)
            events.append(event)
        
        return events
    
    def get_events_in_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Get all events in a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of events in range
        """
        return self.query_events(start_date=start_date, end_date=end_date)
    
    def get_upcoming_events(self, days: int = 7) -> List[Dict]:
        """
        Get upcoming events in the next N days.
        Only returns events that START in the future (not events that are ongoing).
        Includes calendar events and submission deadlines (assignments, quizzes, projects).
        Excludes only job application deadlines.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of upcoming events (including submission deadlines, excluding job application deadlines)
        """
        now = datetime.now()
        future = now + timedelta(days=days)
        
        # For upcoming events, we want events that START in the future
        # Use a direct query instead of query_events which filters by end_time
        # Exclude only job application deadlines, but include submission deadlines
        cur = self.conn.cursor()
        query = """
            SELECT DISTINCT e.*, 
                   GROUP_CONCAT(DISTINCT ee.message_id) as message_ids,
                   COUNT(DISTINCT ee.message_id) as email_count
            FROM events e
            LEFT JOIN event_emails ee ON e.id = ee.event_id
            WHERE e.start_time IS NOT NULL
              AND e.start_time >= ?
              AND e.start_time <= ?
              AND e.summary NOT LIKE '%Job Application Deadline%'
              AND (e.summary NOT LIKE '%Application Deadline%' 
                   OR e.summary LIKE '%Submission%'
                   OR e.summary LIKE '%Assignment%'
                   OR e.summary LIKE '%Quiz%'
                   OR e.summary LIKE '%Project%'
                   OR e.summary LIKE '%Due%'
                   OR e.summary LIKE '%Homework%')
            GROUP BY e.id 
            ORDER BY e.start_time ASC 
            LIMIT ?
        """
        cur.execute(query, (now, future, 50))
        rows = cur.fetchall()
        
        # get column names
        columns = [desc[0] for desc in cur.description]
        
        events = []
        for row in rows:
            event = dict(zip(columns, row))
            # parse message_ids string into list
            if event.get("message_ids"):
                event["message_ids"] = event["message_ids"].split(",")
            else:
                event["message_ids"] = []
            
            # convert datetime strings to datetime objects
            event = self._convert_event_datetimes(event)
            
            # additional filtering: exclude only job application deadlines
            summary = event.get("summary", "").lower()
            # Exclude job application deadlines
            if "job application deadline" in summary:
                continue
            # Allow if it's a submission/assignment/quiz/project deadline
            if "application deadline" in summary:
                # Check if it's an academic deadline (submission, assignment, quiz, project, due, homework)
                if not any(term in summary for term in ["submission", "assignment", "quiz", "project", "due", "homework"]):
                    continue  # It's a job application deadline, exclude it
            events.append(event)
        
        return events
    
    def get_events_this_week(self) -> List[Dict]:
        """Get all events happening this week."""
        now = datetime.now()
        # get start of week (Sunday)
        days_since_sunday = (now.weekday() + 1) % 7
        week_start = now - timedelta(days=days_since_sunday)
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # get end of week (Saturday)
        week_end = week_start + timedelta(days=6)
        week_end = week_end.replace(hour=23, minute=59, second=59)
        
        return self.query_events(start_date=week_start, end_date=week_end)
    
    def search_events(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search events by text query (searches summary, description, location).
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching events
        """
        cur = self.conn.cursor()
        search_term = f"%{query}%"
        
        cur.execute("""
            SELECT * FROM events 
            WHERE summary LIKE ? 
               OR description LIKE ? 
               OR location LIKE ?
            ORDER BY start_time ASC
            LIMIT ?
        """, (search_term, search_term, search_term, limit))
        
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        
        events = []
        for row in rows:
            event = dict(zip(columns, row))
            # convert datetime strings to datetime objects
            event = self._convert_event_datetimes(event)
            events.append(event)
        
        return events
    
    def format_events_for_answer(self, events: List[Dict]) -> str:
        """
        Format events for LLM answer generation.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Formatted string representation
        """
        if not events:
            return "No events found."
        
        formatted = []
        for event in events:
            summary = event.get("summary", "Untitled Event")
            location = event.get("location", "")
            start_str = event.get("start_time_str", "Unknown time")
            end_str = event.get("end_time_str", "")
            organizer = event.get("organizer", "")
            description = event.get("description", "")
            email_count = event.get("email_count", 0)
            
            event_text = f"**{summary}**\n"
            if start_str:
                event_text += f"Time: {start_str}"
                if end_str:
                    event_text += f" - {end_str}"
                event_text += "\n"
            if location:
                event_text += f"Location: {location}\n"
            if organizer:
                event_text += f"Organizer: {organizer}\n"
            if description:
                event_text += f"Description: {description}\n"
            if email_count > 1:
                event_text += f"*Found in {email_count} emails (invite, reminders, updates)*\n"
            
            formatted.append(event_text)
        
        return "\n---\n".join(formatted)
    
    def get_event_emails(self, event_id: int) -> List[Dict]:
        """
        Get all emails that contain a specific event.
        
        Args:
            event_id: Event ID
            
        Returns:
            List of email information dictionaries
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT message_id, attachment_filename, email_date, is_primary
            FROM event_emails
            WHERE event_id = ?
            ORDER BY is_primary DESC, email_date DESC
        """, (event_id,))
        
        rows = cur.fetchall()
        return [
            {
                "message_id": row[0],
                "attachment_filename": row[1],
                "email_date": row[2],
                "is_primary": bool(row[3])
            }
            for row in rows
        ]
    
    def count(self) -> int:
        """Get total number of unique events in database."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]
    
    def get_deduplication_stats(self) -> Dict:
        """
        Get statistics about event deduplication.
        
        Returns:
            Dictionary with deduplication statistics
        """
        cur = self.conn.cursor()
        
        # total unique events
        cur.execute("SELECT COUNT(*) FROM events")
        unique_events = cur.fetchone()[0]
        
        # total event-email links
        cur.execute("SELECT COUNT(*) FROM event_emails")
        total_links = cur.fetchone()[0]
        
        # events with multiple emails
        cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT event_id, COUNT(*) as email_count
                FROM event_emails
                GROUP BY event_id
                HAVING email_count > 1
            )
        """)
        duplicated_events = cur.fetchone()[0]
        
        # avg emails per event
        avg_emails = total_links / unique_events if unique_events > 0 else 0
        
        return {
            "unique_events": unique_events,
            "total_event_email_links": total_links,
            "events_with_multiple_emails": duplicated_events,
            "average_emails_per_event": round(avg_emails, 2),
            "deduplication_rate": round((1 - unique_events / total_links) * 100, 1) if total_links > 0 else 0
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()

