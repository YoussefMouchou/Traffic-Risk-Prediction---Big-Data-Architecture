import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Traffic Predictions Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the UI
st.markdown("""
<style>
    /* Main theme colors and fonts */
    :root {
        --primary-color: #00C4B4;
        --secondary-color: #B0BEC5;
        --text-color: #E0E0E0;
        --background-color: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        --card-background: #263238;
    }

    /* Overall page styling */
    .main {
        background: var(--background-color);
        padding: 2rem;
        min-height: 100vh;
        color: var(--text-color);
        font-family: 'Arial', sans-serif;
    }

    /* Header styling */
    .main h1 {
        color: var(--primary-color);
        font-size: 2.8rem;
        font-weight: 600;
        margin-bottom: 2rem;
        border-bottom: 2px solid var(--primary-color);
                border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }

    .main h2, .main h3 {
        color: var(--primary-color);
        margin-top: 2rem;
        font-weight: 500;
        font-size: 1.6rem;
    }

    /* Card effect for charts */
    .chart-container {
        background: var(--card-background);
        border: 1px solid rgba(0, 196, 180, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        transition: box-shadow 0.3s ease;
    }

    .chart-container:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }
        /* Metric cards */
    .metric-card {
        background: var(--card-background);
        border: 1px solid rgba(0, 196, 180, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        text-align: center;
        transition: box-shadow 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--primary-color);
    }

    .metric-label {
        font-size: 1rem;
        color: var(--secondary-color);
        margin-top: 0.5rem;
    }
    /* Dataframe styling */
    .dataframe-container {
        background: var(--card-background);
        border: 1px solid rgba(0, 196, 180, 0.2);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        overflow-x: auto;
    }

    /* Filters section */
    .filters-section {
        background: var(--card-background);
        border: 1px solid rgba(0, 196, 180, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }

    /* Success message */
    .success {
        padding: 1rem;
        background: rgba(0, 196, 180, 0.1);
        color: var(--primary-color);
        border: 1px solid var(--primary-color);
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
        /* Error message */
    .error {
        padding: 1rem;
        background: rgba(255, 82, 82, 0.1);
        color: #FF5252;
        border: 1px solid #FF5252;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }

    /* Custom button styling */
    .stButton > button {
        background: var(--primary-color);
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 0.8rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background: #00A69A;
    }

    /* Dataframe text styling */
    .dataframe-container table {
        color: var(--text-color);
        background: transparent;
    }"
        /* Responsive adjustments */
    @media (max-width: 1024px) {
        .main {
            padding: 1.5rem;
        }
        .main h1 {
            font-size: 2.2rem;
        }
        .main h2, .main h3 {
            font-size: 1.4rem;
        }
        .metric-value {
            font-size: 1.8rem;
        }
        .metric-card, .chart-container, .filters-section {
            padding: 1rem;
        }
        .stButton > button {
            padding: 0.6rem 1.5rem;
        }
    }

    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        .main h1 {
            font-size: 1.8rem;
        }
        .main h2, .main h3 {
            font-size: 1.2rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
        .metric-label {
            font-size: 0.9rem;
        }
        .metric-card, .chart-container, .filters-section {
                    padding: 0.8rem;
        }
        .stButton > button {
            padding: 0.5rem 1.2rem;
            font-size: 0.9rem;
        }
        /* Stack columns on smaller screens */
        .st-columns > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }

    @media (max-width: 480px) {
        .main {
            padding: 0.5rem;
        }
        .main h1 {
            font-size: 1.5rem;
        }
        .main h2, .main h3 {
            font-size: 1rem;
        }
        .metric-value {
            font-size: 1.2rem;
        }
        .metric-label {
            font-size: 0.8rem;
        }
        .metric-card, .chart-container, .filters-section {
            padding: 0.5rem;
        }
        .stButton > button {
            padding: 0.4rem 1rem;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)
# Page title with enhanced styling
st.markdown("<h1>Traffic Predictions Dashboard</h1>", unsafe_allow_html=True)

# Function to connect to PostgreSQL
def connect_to_postgres():
    try:
        conn = psycopg2.connect(
            host="172.18.0.2",
            port=5432,
            database="traffic_db",
            user="mobility_user",
            password="2005"
        )
        st.markdown('<div class="success">Connected to PostgreSQL!</div>', unsafe_allow_html=True)
        return conn
    except Exception as e:
        st.markdown(f'<div class="error">Failed to connect to PostgreSQL: {e}</div>', unsafe_allow_html=True)
        st.stop()

# Function to fetch data
def fetch_data(conn):
    try:
        query = "SELECT * FROM traffic_predictions;"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.markdown(f'<div class="error">Failed to fetch data: {e}</div>', unsafe_allow_html=True)
        return None

# Connect to PostgreSQL
conn = connect_to_postgres()
# Fetch data
df = fetch_data(conn)
conn.close()
if df is not None:
    # Display dashboard metrics
    st.markdown("<h2>Dashboard Overview</h2>", unsafe_allow_html=True)

    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Average Vehicle Count</div>
        </div>
        """.format(round(df['vehicle_count'].mean(), 1)), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Max Prediction</div>
        </div>
        """.format(round(df['prediction'].max(), 2)), unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Unique Locations</div>
        </div>
        """.format(df.groupby(['latitude', 'longitude']).ngroups), unsafe_allow_html=True)
            # Convert timestamp
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')

    # Raw Data section with styled container
    st.markdown("<h2>Raw Data</h2>", unsafe_allow_html=True)
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df, height=300)
    st.markdown('</div>', unsafe_allow_html=True)

    # Create two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3>Vehicle Count Over Time</h3>", unsafe_allow_html=True)
        fig = px.line(df, x='timestamp', y='vehicle_count',
                      title=None,
                      line_shape='spline',
                      template='plotly_dark')
        fig.update_traces(line=dict(color='#00C4B4', width=3))
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Vehicle Count",
            legend_title="Legend",
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("<h3>Traffic Conditions Distribution</h3>", unsafe_allow_html=True)
            fig = px.histogram(df, x='traffic_condition',
                           title=None,
                           color='traffic_condition',
                           color_discrete_sequence=['#00C4B4', '#4DD0E1', '#80DEEA', '#B0BEC5'],
                           template='plotly_dark')
        fig.update_layout(
            xaxis_title="Traffic Condition",
            yaxis_title="Count",
            showlegend=False,
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interactive filters section
    st.markdown("<h2>Interactive Filters</h2>", unsafe_allow_html=True)
    st.markdown('<div class="filters-section">', unsafe_allow_html=True)

    # Create filters in columns
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        # Traffic condition filter
        traffic_conditions = df['traffic_condition'].unique()
        selected_condition = st.selectbox("Select Traffic Condition", traffic_conditions)

    with filter_col2:
        # Time range filter
        start_time = st.date_input("Start Date", value=datetime.now().replace(day=1))
        end_time = st.date_input("End Date", value=datetime.now())
            # Prediction threshold slider
    min_prediction = float(df['prediction'].min())
    max_prediction = float(df['prediction'].max())
    prediction_threshold = st.slider("Prediction Threshold",
                                    min_value=min_prediction,
                                    max_value=max_prediction,
                                    value=(min_prediction + max_prediction) / 2)

    st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    filtered_df = df[df['traffic_condition'] == selected_condition]
    filtered_df_time = df[(df['timestamp'].dt.date >= start_time) & (df['timestamp'].dt.date <= end_time)]
    filtered_df_prediction = df[df['prediction'] > prediction_threshold]

    # Filtered visualizations
    st.markdown("<h2>Filtered Visualizations</h2>", unsafe_allow_html=True)

    # Create two columns for filtered charts
    fcol1, fcol2 = st.columns(2)

    with fcol1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3>Traffic Predictions by Location</h3>", unsafe_allow_html=True)
        fig = px.scatter(filtered_df, x='longitude', y='latitude',
                         color='prediction', size='vehicle_count',
                         hover_data=['traffic_condition'],
                         color_continuous_scale='Teal',
                         title=None,
                         template='plotly_dark')
        fig.update_layout(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with fcol2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3>Vehicle Count Over Selected Time Range</h3>", unsafe_allow_html=True)
        fig = px.line(filtered_df_time, x='timestamp', y='vehicle_count',
                      title=None,
                      line_shape='spline',
                      template='plotly_dark')
        fig.update_traces(line=dict(color='#00C4B4', width=3))
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Vehicle Count",
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Predictions above threshold
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("<h3>Traffic Predictions Above Threshold</h3>", unsafe_allow_html=True)
    fig = px.scatter_mapbox(filtered_df_prediction,
                           lat='latitude',
                           lon='longitude',
                           color='prediction',
                           size='vehicle_count',
                           hover_data=['traffic_condition'],
                           color_continuous_scale='Teal',
                           zoom=10,
                           mapbox_style="carto-darkmatter",
                           title=None)
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        height=500,
        font=dict(color='#E0E0E0')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)