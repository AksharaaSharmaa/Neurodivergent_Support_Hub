import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from streamlit_calendar import calendar
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pytz
from google.cloud import secretmanager
import google.auth
import google.generativeai as genai
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Back button
if st.button("← Back to Home"):
    st.switch_page("HomePage.py")

# Function to access secrets from Google Secret Manager
def access_secret(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    project_id = "neurodivai"
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})
    return response.payload.data.decode("UTF-8")

# Add Gemini API configuration
genai.configure(api_key=access_secret("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_calendar_service():
    credentials_info = json.loads(access_secret("SERVICE_ACCOUNT_JSON"))
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info, scopes=SCOPES
    )
    return build('calendar', 'v3', credentials=credentials)

SCOPES = ['https://www.googleapis.com/auth/calendar']

# Add neurodivergent conditions and priority levels
CONDITIONS = ["ADHD", "Autism", "Dyslexia", "Executive Function Disorder", "Sensory Processing Disorder"]
PRIORITY_LEVELS = ["High", "Medium", "Low"]

def get_schedule_analysis(events, condition):
    events_text = "\n".join([
        f"Event: {e['summary']}, Date: {e['start'].get('dateTime', e['start'].get('date'))}, "
        f"Priority: {e.get('extendedProperties', {}).get('private', {}).get('priority', 'Not set')}, "
        f"Description: {e.get('description', 'No description')}"
        for e in events
    ])
    
    prompt = f"""As a schedule management expert specializing in {condition}, analyze this monthly schedule:

{events_text}

Provide specific advice on:
1. Potential challenges for someone with {condition}
2. Schedule optimization suggestions
3. Coping strategies for task management
Keep response concise and practical."""

    response = model.generate_content(prompt)
    return response.text

SCOPES = ['https://www.googleapis.com/auth/calendar']


def create_event(service, summary, start_time, end_time, description="", location="", priority="Medium"):
    # Ensure timezone is UTC for Google Calendar
    if start_time.tzinfo is None:
        start_time = pytz.UTC.localize(start_time)
    else:
        start_time = start_time.astimezone(pytz.UTC)
        
    if end_time.tzinfo is None:
        end_time = pytz.UTC.localize(end_time)
    else:
        end_time = end_time.astimezone(pytz.UTC)
    
    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'UTC',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'UTC',
        },
        'extendedProperties': {
            'private': {
                'priority': priority
            }
        }
    }
    return service.events().insert(calendarId='primary', body=event).execute()

def delete_event(service, event_id):
    service.events().delete(calendarId='primary', eventId=event_id).execute()

def get_events(service):
    now = datetime.utcnow().isoformat() + 'Z'
    events_result = service.events().list(
        calendarId='primary',
        timeMin=now,
        maxResults=10,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    return events_result.get('items', [])

def format_events_for_calendar(events):
    calendar_events = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        
        # Parse the datetime strings
        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace('Z', '+00:00'))
            
        calendar_events.append({
            'id': event.get('id', ''),
            'title': event.get('summary', 'No Title'),
            'start': start.isoformat(),
            'end': end.isoformat(),
        })
    return calendar_events

def get_schedule_metrics(events):
    if not events:
        return pd.DataFrame()
    
    data = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        
        duration = (datetime.fromisoformat(event['end'].get('dateTime', event['end'].get('date')).replace('Z', '+00:00')) 
                   - datetime.fromisoformat(event['start'].get('dateTime', event['start'].get('date')).replace('Z', '+00:00')))
        
        priority = event.get('extendedProperties', {}).get('private', {}).get('priority', 'Not set')
        
        data.append({
            'date': start.date(),
            'hour': start.hour,
            'duration': duration.total_seconds() / 3600,
            'priority': priority,
            'summary': event['summary']
        })
    
    return pd.DataFrame(data)

def custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: #e2e8f0;
        }
        .calendar-card {
            background-color: #2d2d2d;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .event-card {
            background: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.75rem 0;
            transition: all 0.3s;
        }
        .event-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            transform: translateY(-2px);
        }
        .calendar-header {
            color: #60a5fa;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .form-container {
            background: #2d2d2d;
            padding: 1.8rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .analysis-section {
            margin-top: 2rem;
            padding: 0;
            width: 100%;
        }
        .analysis-card {
            background: #2d2d2d;
            border-radius: 12px;
            padding: 1.8rem;
            margin-top: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            width: 100%;
        }
        .stButton button {
            background-color: #3b82f6;
            color: white;
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            border: none;
            transition: all 0.3s;
            font-weight: 600;
        }
        .stButton button:hover {
            background-color: #2563eb;
            transform: translateY(-2px);
        }
        .delete-button {
            background-color: #dc2626 !important;
        }
        .delete-button:hover {
            background-color: #b91c1c !important;
        }
        .fc {
            background: #2d2d2d;
            border-radius: 12px;
            padding: 1rem;
        }
        .fc-theme-standard td, .fc-theme-standard th {
            border-color: #404040;
        }
        .fc-theme-standard .fc-scrollgrid {
            border-color: #404040;
        }
        .fc-day-today {
            background: #374151 !important;
        }
        .fc-event {
            background: #3b82f6;
            border: none;
            border-radius: 6px;
            padding: 2px 4px;
        }
        .fc-toolbar-title {
            color: #e2e8f0;
        }
        .fc button {
            background: #4b5563;
            color: #e2e8f0;
        }
        .stTextInput input, .stTextArea textarea {
            background: #1f1f1f;
            color: #e2e8f0;
            border-color: #404040;
        }
        .stDateInput input, .stTimeInput input, .stNumberInput input {
            background: #1f1f1f;
            color: #e2e8f0;
            border-color: #404040;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    page_title = "NeuroSchedule - Task Manager for Neurodivergent Minds"
    custom_css()
    
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #60a5fa; font-size: 2.5rem; font-weight: 800;'>🧠 NeuroSchedule</h1>
            <p style='color: #9ca3af; font-size: 1.2rem;'>Task Management Designed for Neurodivergent Minds</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        service = get_calendar_service()
        
        # Main content area
        col1, col2, col3 = st.columns([2,1,1])
        
        # Column 1: Calendar
        with col1:
            st.markdown('<div class="calendar-header">📅 Visual Schedule</div>', unsafe_allow_html=True)
            events = get_events(service)
            calendar_events = format_events_for_calendar(events)
            calendar_options = {
                "initialView": "dayGridMonth",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "dayGridMonth,timeGridWeek,timeGridDay"
                },
                "height": 650,
                "slotMinTime": "06:00:00",
                "slotMaxTime": "22:00:00",
                "events": calendar_events,
                "selectable": True,
                "editable": True,
                "nowIndicator": True,
                "dayMaxEvents": 3,
                "eventTimeFormat": {
                    "hour": "2-digit",
                    "minute": "2-digit",
                    "meridiem": True
                }
            }
            selected_event = calendar(events=calendar_events, options=calendar_options)

        # Column 2: New Task Form
        with col2:
            st.markdown('<div class="calendar-header">➕ New Task</div>', unsafe_allow_html=True)
            with st.form("event_form", clear_on_submit=True):
                summary = st.text_input("Task Name")
                description = st.text_area("Details & Notes")
                location = st.text_input("Location (Optional)")
                priority = st.select_slider("Priority", options=PRIORITY_LEVELS, value="Medium")
                
                col_date, col_time = st.columns(2)
                with col_date:
                    event_date = st.date_input("Date")
                with col_time:
                    start_time = st.time_input("Start Time")
                    duration = st.number_input("Duration (hours)", min_value=0.5, value=1.0, step=0.5)
                
                # Create naive datetime objects first
                start_datetime = datetime.combine(event_date, start_time)
                end_datetime = start_datetime + timedelta(hours=duration)
                
                submitted = st.form_submit_button("Add to Schedule")
                if submitted and summary:
                    with st.spinner("Adding task..."):
                        create_event(service, summary, start_datetime, end_datetime, 
                                   description, location, priority)
                        st.success("Task added successfully!")
                        st.rerun()

        # Column 3: Upcoming Tasks
        with col3:
            st.markdown('<div class="calendar-header">📋 Upcoming Tasks</div>', 
                       unsafe_allow_html=True)
            if not events:
                st.info("No upcoming tasks scheduled")
            
            for event in events:
                with st.container():
                    start_time = event['start'].get('dateTime', event['start'].get('date'))
                    if isinstance(start_time, str):
                        # Parse the datetime string
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        # Format for display
                        formatted_time = start_time.astimezone(pytz.UTC).strftime("%B %d, %Y at %I:%M %p UTC")
                    
                    priority = event.get('extendedProperties', {}).get('private', {}).get('priority', 'Not set')
                    priority_color = {
                        'High': '#ef4444',
                        'Medium': '#f59e0b',
                        'Low': '#22c55e'
                    }.get(priority, '#9ca3af')
                    
                    st.markdown(f"""
                    <div class="event-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="color: #60a5fa; margin: 0 0 0.5rem 0; font-size: 1.2rem;">{event['summary']}</h3>
                            <span style="background-color: {priority_color}; padding: 2px 8px; border-radius: 12px; color: white; font-size: 0.8rem;">
                                {priority}
                            </span>
                        </div>
                        <p style="color: #9ca3af; margin: 0.25rem 0;">
                            <strong>📅</strong> {formatted_time}
                        </p>
                        <p style="color: #9ca3af; margin: 0.25rem 0;">
                            <strong>📍</strong> {event.get('location', 'No location')}
                        </p>
                        <p style="color: #9ca3af; margin: 0.25rem 0;">
                            <strong>📝</strong> {event.get('description', 'No description')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button('🗑️ Remove', key=event['id'], type="secondary"):
                        with st.spinner("Removing task..."):
                            delete_event(service, event['id'])
                            st.success("Task removed successfully!")
                            st.rerun()

        # Analysis section remains the same
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        selected_condition = st.selectbox("Analyze schedule for:", options=CONDITIONS)
        
        if st.button("Analyze Schedule"):
            with st.spinner(f"Analyzing schedule for {selected_condition}..."):
                analysis = get_schedule_analysis(events, selected_condition)
                st.markdown(f"""
                <div class="analysis-card">
                    <h3 style="color: #60a5fa;">Schedule Analysis for {selected_condition}</h3>
                    <p style="color: #e2e8f0; white-space: pre-line;">{analysis}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your API credentials and try again.")
        
if __name__ == "__main__":
    main()