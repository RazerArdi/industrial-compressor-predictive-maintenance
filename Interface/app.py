import os
import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

external_stylesheets = ['https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Company Asset Guard"

model_path = os.path.join(os.path.dirname(__file__), '../Model/xgb_compressor_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), '../Model/model_features.pkl')

try:
    model = joblib.load(model_path)
    model_features = joblib.load(features_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_features = []

THEME = {
    'background': '#0a0e1a',
    'card_bg': '#151b2e',
    'card_hover': '#1a2235',
    'text': '#e2e8f0',
    'subtext': '#94a3b8',
    'accent': '#3b82f6',
    'accent_light': '#60a5fa',
    'danger': '#ef4444',
    'danger_light': '#f87171',
    'warning': '#f59e0b',
    'success': '#10b981',
    'success_light': '#34d399',
    'grid': '#1e293b',
    'border': '#334155'
}

INFO_ICON_STYLE = {
    'display': 'inline-flex',
    'alignItems': 'center',
    'justifyContent': 'center',
    'width': '20px',
    'height': '20px',
    'borderRadius': '50%',
    'backgroundColor': 'rgba(59, 130, 246, 0.1)',
    'backdropFilter': 'blur(10px)',
    'border': f'1.5px solid {THEME["accent"]}',
    'color': THEME['accent'],
    'fontSize': '12px',
    'fontWeight': '700',
    'fontFamily': 'Inter, sans-serif',
    'marginLeft': '8px',
    'cursor': 'help',
    'boxShadow': '0 4px 15px rgba(59, 130, 246, 0.2)',
    'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
}


def clean_column_names(df):
    df_clean = df.copy()
    col_map = {
        'Motor_Current': 'Motor_current',
        'H1': 'Oil_temperature',
        'dv_pressure': 'DV_pressure',
        'reservoirs': 'Reservoirs',
        'tp2': 'TP2',
        'tp3': 'TP3'
    }
    df_clean = df_clean.rename(columns=col_map)
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
    return df_clean

def generate_physics_features(df):
    df_processed = clean_column_names(df)
    
    defaults = {'Motor_current': 0.0, 'Oil_temperature': 60.0, 'TP2': 0.0}
    for col, val in defaults.items():
        if col not in df_processed.columns:
            df_processed[col] = val

    df_processed['efficiency_index'] = df_processed['TP2'] / (df_processed['Motor_current'] + 0.1)
    df_processed['thermal_efficiency'] = df_processed['TP2'] / (df_processed['Oil_temperature'] + 0.1)
    
    if len(df_processed) > 1:
        df_processed['TP2_volatility'] = df_processed['TP2'].rolling(window=20).std().fillna(0)
        df_processed['Motor_volatility'] = df_processed['Motor_current'].rolling(window=20).std().fillna(0)
    else:
        eff_val = df_processed['efficiency_index'].values[0]
        simulated_vol = 0.05 + max(0, (2.0 - eff_val) * 2.5) 
        df_processed['TP2_volatility'] = simulated_vol
        df_processed['Motor_volatility'] = simulated_vol
        
    for col in model_features:
        if col not in df_processed.columns:
            df_processed[col] = 0.0
            
    return df_processed[model_features]

def create_header_with_info(label, info_text, align='left'):
    return html.Div([
        html.Span(label, style={
            'color': THEME['text'], 
            'fontWeight': '600', 
            'fontSize': '1.1rem',
            'fontFamily': 'Inter, sans-serif',
            'letterSpacing': '-0.025em'
        }),
        html.Div("i", title=info_text, style=INFO_ICON_STYLE)
    ], style={
        'display': 'flex', 
        'alignItems': 'center', 
        'justifyContent': align, 
        'marginBottom': '15px'
    })

def create_control_card(id, label, unit, min_val, max_val, default, info_text, range_text):
    return html.Div([
        create_header_with_info(f"{label} ({unit})", info_text, align='space-between'),
        html.Div(range_text, style={
            'fontSize': '0.8rem', 
            'color': THEME['subtext'], 
            'marginBottom': '12px', 
            'fontFamily': 'Inter, sans-serif',
            'fontWeight': '400'
        }),
        dcc.Slider(
            id=id, min=min_val, max=max_val, step=0.1, value=default,
            marks={
                min_val: {'label': str(min_val), 'style': {'color': THEME['subtext'], 'fontSize': '0.75rem'}}, 
                max_val: {'label': str(max_val), 'style': {'color': THEME['subtext'], 'fontSize': '0.75rem'}}
            },
            tooltip={"placement": "bottom", "always_visible": True},
            className='modern-slider'
        )
    ], style={
        'marginBottom': '28px', 
        'paddingBottom': '20px',
        'borderBottom': f'1px solid {THEME["border"]}',
        'transition': 'all 0.3s ease'
    })

def create_gauge(probability):
    val = max(0, min(100, probability * 100))
    
    if val < 50:
        color = THEME['success']
        bg_color = 'rgba(16, 185, 129, 0.1)'
    elif val < 86:
        color = THEME['warning']
        bg_color = 'rgba(245, 158, 11, 0.1)'
    else:
        color = THEME['danger']
        bg_color = 'rgba(239, 68, 68, 0.1)'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "", 'font': {'size': 1}},
        number={
            'suffix': "%", 
            'font': {
                'color': color, 
                'size': 48,
                'family': 'Inter, sans-serif',
                'weight': 700
            }
        },
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 2, 
                'tickcolor': THEME['subtext'],
                'tickfont': {'size': 10, 'color': THEME['subtext']}
            },
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [50, 86], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [86, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': THEME['text'], 'width': 4}, 
                'thickness': 0.8, 
                'value': 86
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        font={'color': THEME['text'], 'family': 'Inter, sans-serif'}, 
        margin=dict(l=20, r=20, t=10, b=10), 
        height=240
    )
    return fig


sidebar_style = {
    'position': 'fixed', 
    'top': 0, 
    'left': 0, 
    'bottom': 0, 
    'width': '280px',
    'padding': '2.5rem 1.5rem',
    'background': f'linear-gradient(180deg, {THEME["card_bg"]} 0%, #0f1629 100%)',
    'borderRight': f'1px solid {THEME["border"]}',
    'boxShadow': '4px 0 24px rgba(0, 0, 0, 0.4)',
    'zIndex': '1000',
    'fontFamily': 'Inter, sans-serif'
}

content_style = {
    'marginLeft': '310px',  
    'padding': '3rem',      
    'backgroundColor': THEME['background'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, sans-serif',
    'transition': 'margin-left 0.3s ease'
}

nav_button_style = {
    'width': '100%',
    'padding': '14px 20px',
    'marginBottom': '12px',
    'backgroundColor': 'transparent',
    'color': THEME['text'],
    'border': f'1.5px solid {THEME["border"]}',
    'borderRadius': '12px',
    'cursor': 'pointer',
    'fontSize': '0.95rem',
    'fontWeight': '500',
    'fontFamily': 'Inter, sans-serif',
    'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    'textAlign': 'left',
    'letterSpacing': '-0.015em'
}

nav_button_active_style = {
    **nav_button_style,
    'backgroundColor': THEME['accent'],
    'borderColor': THEME['accent'],
    'boxShadow': f'0 4px 20px rgba(59, 130, 246, 0.4)',
    'transform': 'translateY(-2px)'
}

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {
                font-family: 'Inter', sans-serif !important;
            }
            
            .modern-slider .rc-slider-track {
                background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
                height: 6px;
            }
            
            .modern-slider .rc-slider-rail {
                background: #1e293b;
                height: 6px;
            }
            
            .modern-slider .rc-slider-handle {
                border: 3px solid #3b82f6;
                background: #0a0e1a;
                width: 20px;
                height: 20px;
                margin-top: -7px;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }
            
            .modern-slider .rc-slider-handle:hover {
                border-color: #60a5fa;
                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
            }
            
            .card-hover:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
            }
            
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #0a0e1a;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #334155;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #475569;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    style={'backgroundColor': THEME['background'], 'minHeight': '100vh'},
    children=[
        
        dcc.Location(id='url', refresh=False),
        
        html.Div(style=sidebar_style, children=[
            html.Div([
                html.Div(style={
                    'width': '60px',
                    'height': '60px',
                    'margin': '0 auto 15px',
                    'background': 'linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)',
                    'borderRadius': '16px',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'boxShadow': '0 8px 32px rgba(59, 130, 246, 0.3)',
                    'fontSize': '1.8rem',
                    'fontWeight': 'bold',
                    'color': 'white'
                }, children=['AG']),
                html.H2("ASSET GUARD", style={
                    'color': THEME['text'],
                    'textAlign': 'center',
                    'fontSize': '1.5rem',
                    'fontWeight': '800',
                    'letterSpacing': '-0.05em',
                    'marginBottom': '5px',
                    'background': 'linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent'
                }),
                html.P("Predictive Maintenance AI", style={
                    'textAlign': 'center',
                    'fontSize': '0.8rem',
                    'color': THEME['subtext'],
                    'fontWeight': '400',
                    'marginBottom': '0'
                })
            ], style={'marginBottom': '40px'}),
            
            html.Div(style={
                'height': '1px',
                'background': f'linear-gradient(90deg, transparent 0%, {THEME["border"]} 50%, transparent 100%)',
                'marginBottom': '30px'
            }),
            
            html.Div([
                html.P("NAVIGATION", style={
                    'fontSize': '0.7rem',
                    'color': THEME['subtext'],
                    'fontWeight': '700',
                    'letterSpacing': '0.1em',
                    'marginBottom': '15px',
                    'textTransform': 'uppercase'
                }),
                dcc.Link(
                    html.Button('Real-Time Monitor', id='nav-realtime', style=nav_button_style),
                    href='/'
                ),
                dcc.Link(
                    html.Button('Batch Analysis', id='nav-batch', style=nav_button_style),
                    href='/batch'
                ),
            ]),
            
            html.Div([
                html.Div(style={
                    'height': '1px',
                    'background': f'linear-gradient(90deg, transparent 0%, {THEME["border"]} 50%, transparent 100%)',
                    'marginTop': '40px',
                    'marginBottom': '20px'
                }),
                html.P("Powered by XGBoost ML", style={
                    'fontSize': '0.75rem',
                    'color': THEME['subtext'],
                    'textAlign': 'center',
                    'marginBottom': '5px'
                }),
                html.P("© 2025 Company Shipping", style={
                    'fontSize': '0.7rem',
                    'color': THEME['subtext'],
                    'textAlign': 'center'
                })
            ], style={'position': 'absolute', 'bottom': '30px', 'left': '1.5rem', 'right': '1.5rem'})
        ]),
        
        html.Div(id='page-content', style=content_style)
    ]
)

card_style = {
    'backgroundColor': THEME['card_bg'],
    'padding': '28px',
    'borderRadius': '20px',
    'boxShadow': '0 4px 24px rgba(0, 0, 0, 0.4)',
    'border': f'1px solid {THEME["border"]}',
    'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    'className': 'card-hover'
}

realtime_layout = html.Div([
    html.Div([
        html.H1("Real-Time Diagnostics", style={
            'color': THEME['text'],
            'fontSize': '2.5rem',
            'fontWeight': '800',
            'marginBottom': '8px',
            'letterSpacing': '-0.05em'
        }),
        html.P("Live telemetry monitoring with AI-powered anomaly detection", style={
            'color': THEME['subtext'],
            'fontSize': '1rem',
            'marginBottom': '35px',
            'fontWeight': '400'
        })
    ]),
    
    html.Div(style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}, children=[  
        html.Div(style={
            **card_style,
            'flex': '1',
            'minWidth': '340px',
            'maxWidth': '450px'
        }, children=[
            html.Div([
                html.H4("Telemetry Parameters", style={
                    'color': THEME['text'],
                    'fontSize': '1.3rem',
                    'fontWeight': '700',
                    'marginBottom': '8px',
                    'letterSpacing': '-0.025em'
                }),
                html.P("Adjust sensor inputs to simulate scenarios", style={
                    'color': THEME['subtext'],
                    'fontSize': '0.85rem',
                    'marginBottom': '25px'
                })
            ]),
            create_control_card('input-tp2', 'Air Pressure (TP2)', 'bar', 0, 15, 8.5, 
                "Outlet pressure measurement", "Nominal: 8.0-10.5 | Critical: <6.0"),
            create_control_card('input-tp3', 'Oil Pressure (TP3)', 'bar', 0, 12, 7.0,
                "Lubrication system pressure", "Nominal: 6.5-8.0 | Critical: <5.0"),
            create_control_card('input-h1', 'Oil Temperature', '°C', 20, 120, 65.0,
                "Oil cooling temperature", "Nominal: 55-75 | Critical: >90"),
            create_control_card('input-motor', 'Motor Current', 'A', 0, 12, 3.5,
                "Motor electrical load", "Nominal: 3.5-6.0 | Critical: >9.0"),
            create_control_card('input-res', 'Reservoir Pressure', 'bar', 0, 15, 8.5,
                "Storage tank pressure", "Should match TP2 ±0.5"),
        ]),
        
        html.Div(style={'flex': '2', 'minWidth': '500px', 'display': 'flex', 'flexDirection': 'column', 'gap': '24px'}, children=[
            
            html.Div(style={**card_style, 'height': '340px'}, children=[
                create_header_with_info("Failure Probability", 
                    "AI prediction with physics-based feature engineering", align='center'),
                dcc.Graph(id='gauge-chart', config={'displayModeBar': False}, style={'height': '270px'})
            ]),
            
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '20px'}, children=[
                html.Div(style={
                    **card_style,
                    'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, #1a2235 100%)',
                    'borderLeft': f'4px solid {THEME["success"]}'
                }, children=[
                    create_header_with_info("Efficiency Index", "Pressure output per unit current", align='left'),
                    html.H2(id='efficiency-val', style={'color': THEME['text'], 'fontSize': '2.2rem', 'fontWeight': '800', 'margin': '10px 0'}),
                    html.Div(id='efficiency-desc', style={'fontSize': '0.85rem', 'color': THEME['subtext'], 'fontWeight': '500'})
                ]),
                
                html.Div(style={
                    **card_style,
                    'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, #1a2235 100%)',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'borderLeft': f'4px solid {THEME["accent"]}'
                }, children=[
                    create_header_with_info("System Status", "Decision threshold: 86%", align='center'),
                    html.Div(id='status-text-container', style={'marginTop': '15px'})
                ])
            ]),
            
            html.Div(style={
                **card_style,
                'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, #1a2235 100%)',
                'borderLeft': f'4px solid {THEME["warning"]}'
            }, children=[
                create_header_with_info("Diagnostic Insight", "Automated root cause analysis", align='left'),
                html.P(id='diagnostic-text', style={
                    'color': THEME['text'],
                    'fontSize': '1rem',
                    'lineHeight': '1.6',
                    'fontWeight': '500',
                    'marginTop': '10px'
                })
            ])
        ])
    ])
])

batch_layout = html.Div([
    html.Div([
        html.H1("Batch Analysis & Validation", style={
            'color': THEME['text'],
            'fontSize': '2.5rem',
            'fontWeight': '800',
            'marginBottom': '8px',
            'letterSpacing': '-0.05em'
        }),
        html.P("Upload historical data for comprehensive failure trend analysis", style={
            'color': THEME['subtext'],
            'fontSize': '1rem',
            'marginBottom': '35px',
            'fontWeight': '400'
        })
    ]),
    
    html.Div(style={
        **card_style,
        'marginBottom': '24px',
        'border': f'2px dashed {THEME["accent"]}',
        'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, rgba(59, 130, 246, 0.05) 100%)'
    }, children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.Div(style={
                    'width': '80px',
                    'height': '80px',
                    'margin': '0 auto 20px',
                    'border': f'3px solid {THEME["accent"]}',
                    'borderRadius': '50%',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'fontSize': '2rem',
                    'color': THEME['accent']
                }, children=['CSV']),
                html.Div([
                    'Drag and Drop your CSV file here or ',
                    html.A('Browse Files', style={
                        'color': THEME['accent'],
                        'fontWeight': '700',
                        'textDecoration': 'underline',
                        'cursor': 'pointer'
                    })
                ], style={'fontSize': '1rem', 'color': THEME['text']}),
                html.P('Supported format: CSV with sensor telemetry data', style={
                    'fontSize': '0.85rem',
                    'color': THEME['subtext'],
                    'marginTop': '10px'
                })
            ], style={'textAlign': 'center', 'padding': '30px'}),
            style={'width': '100%', 'minHeight': '180px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'},
            multiple=False
        ),
        html.Div(id='upload-status', style={
            'marginTop': '15px',
            'textAlign': 'center',
            'color': THEME['accent'],
            'fontSize': '0.95rem',
            'fontWeight': '600'
        })
    ]),
    
    dcc.Loading(
        id="loading-batch",
        type="circle",
        color=THEME['accent'],
        children=[html.Div(id='batch-results-container')]
    )
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    return batch_layout if pathname == '/batch' else realtime_layout

@app.callback(
    [Output('gauge-chart', 'figure'),
     Output('efficiency-val', 'children'),
     Output('efficiency-val', 'style'),
     Output('efficiency-desc', 'children'),
     Output('status-text-container', 'children'),
     Output('diagnostic-text', 'children')],
    [Input('input-tp2', 'value'),
     Input('input-tp3', 'value'),
     Input('input-h1', 'value'),
     Input('input-motor', 'value'),
     Input('input-res', 'value')]
)
def update_realtime(tp2, tp3, h1, motor, res):
    raw_data = {
        'TP2': [float(tp2)],
        'TP3': [float(tp3)],
        'Oil_temperature': [float(h1)],
        'Motor_current': [float(motor)],
        'Reservoirs': [float(res)]
    }
    df = pd.DataFrame(raw_data)
    
    try:
        X = generate_physics_features(df)
        proba = model.predict_proba(X)[0][1] if model else 0.0
        
        gauge = create_gauge(proba)
        
        eff = X['efficiency_index'].values[0]
        eff_color = THEME['success'] if eff > 1.5 else (THEME['warning'] if eff > 1.0 else THEME['danger'])
        eff_val_text = f"{eff:.2f}"
        
        if eff > 1.8:
            eff_desc = "Optimal Performance"
        elif eff > 1.2:
            eff_desc = "Moderate Efficiency"
        else:
            eff_desc = "Low Efficiency (Energy Loss)"
        
        is_critical = proba >= 0.86
        status_text = "CRITICAL" if is_critical else "NORMAL"
        status_color = THEME['danger'] if is_critical else THEME['success']
        status_div = html.H2(status_text, style={
            'color': status_color,
            'fontWeight': '800',
            'margin': 0,
            'fontSize': '1.8rem',
            'letterSpacing': '-0.025em'
        })
        
        if is_critical:
            if eff < 1.2:
                diag_msg = "ALERT: Critical air leak detected! High power consumption with low pressure output indicates severe inefficiency."
            elif float(h1) > 90:
                diag_msg = "ALERT: Thermal overload! Oil temperature exceeding safety limits - immediate cooling system check required."
            elif float(tp2) < 6.0:
                diag_msg = "ALERT: Compressor failure imminent! Output pressure has collapsed below operational threshold."
            else:
                diag_msg = "ALERT: Complex anomaly pattern detected. Multiple sensor deviations require immediate system inspection."
        elif proba > 0.5:
            diag_msg = "WARNING: Performance degradation detected. System metrics deviating from baseline - schedule preventive maintenance."
        else:
            diag_msg = "System operating within nominal parameters. All sensors reporting healthy values."
            
        return gauge, eff_val_text, {'color': eff_color}, eff_desc, status_div, diag_msg
        
    except Exception as e:
        return go.Figure(), "Error", {}, "", html.Div(), f"Calculation Error: {str(e)}"

@app.callback(
    [Output('batch-results-container', 'children'),
     Output('upload-status', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_batch(contents, filename):
    if contents is None:
        return html.Div(), ""
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        X_full = generate_physics_features(df)
        
        if model:
            probs_full = model.predict_proba(X_full)[:, 1]
        else:
            probs_full = np.zeros(len(X_full))
            
        df['Failure_Probability'] = probs_full
        df['Status'] = np.where(probs_full > 0.86, 'CRITICAL', 'NORMAL')
        
        if len(df) > 5000:
            step = len(df) // 2000
            df_display = df.iloc[::step].copy()
            msg = f"SUCCESS: {filename} processed ({len(df):,} records) - Chart optimized for performance"
        else:
            df_display = df.copy()
            msg = f"SUCCESS: {filename} loaded successfully ({len(df):,} records)"
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display['Failure_Probability'],
            mode='lines',
            name='Failure Probability',
            line=dict(color=THEME['accent'], width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>Index:</b> %{x}<br><b>Probability:</b> %{y:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[df_display.index.min(), df_display.index.max()],
            y=[0.86, 0.86],
            mode='lines',
            name='Critical Threshold (86%)',
            line=dict(color=THEME['danger'], width=3, dash='dash'),
            hovertemplate='<b>Threshold:</b> 86%<extra></extra>'
        ))
        
        critical_mask = df_display['Failure_Probability'] > 0.86
        if critical_mask.any():
            fig.add_trace(go.Scatter(
                x=df_display.index[critical_mask],
                y=df_display['Failure_Probability'][critical_mask],
                mode='markers',
                name='Critical Points',
                marker=dict(
                    color=THEME['danger'],
                    size=8,
                    symbol='x',
                    line=dict(width=2, color=THEME['danger_light'])
                ),
                hovertemplate='<b>CRITICAL</b><br>Index: %{x}<br>Prob: %{y:.2%}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': "Failure Probability Time-Series Analysis",
                'font': {'size': 20, 'color': THEME['text'], 'family': 'Inter, sans-serif', 'weight': 700},
                'x': 0.5,
                'xanchor': 'center'
            },
            paper_bgcolor=THEME['card_bg'],
            plot_bgcolor=THEME['background'],
            font={'color': THEME['text'], 'family': 'Inter, sans-serif'},
            hovermode="x unified",
            xaxis=dict(
                showgrid=True,
                gridcolor=THEME['grid'],
                title="Time Index",
                title_font=dict(size=14, weight=600),
                linecolor=THEME['border']
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=THEME['grid'],
                title="Failure Probability",
                title_font=dict(size=14, weight=600),
                tickformat='.0%',
                linecolor=THEME['border']
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor=THEME['border'],
                borderwidth=1
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        critical_count = len(df[df['Status'] == 'CRITICAL'])
        critical_pct = (critical_count / len(df)) * 100
        avg_prob = df['Failure_Probability'].mean() * 100
        max_prob = df['Failure_Probability'].max() * 100

        results = html.Div([
            
            html.Div(style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))',
                'gap': '20px',
                'marginBottom': '24px'
            }, children=[
                
                html.Div(style={
                    **card_style,
                    'borderLeft': f'5px solid {THEME["accent"]}',
                    'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, rgba(59, 130, 246, 0.05) 100%)'
                }, children=[
                    html.Div(style={
                        'width': '45px', 'height': '45px', 'borderRadius': '12px', 'margin': '0 auto 15px',
                        'background': 'rgba(59, 130, 246, 0.2)',
                        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 
                        'color': THEME['accent'], 'fontWeight': 'bold', 'fontSize': '1.1rem'
                    }, children=['DB']),
                    create_header_with_info("Total Records", "Total rows processed from CSV", align='center'),
                    html.H3(f"{len(df):,}", style={'color': THEME['text'], 'margin': 0, 'textAlign': 'center', 'fontSize': '1.8rem', 'fontWeight': '800'})
                ]),

                html.Div(style={
                    **card_style,
                    'borderLeft': f'5px solid {THEME["danger"]}',
                    'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, rgba(239, 68, 68, 0.05) 100%)'
                }, children=[
                    html.Div(style={
                        'width': '45px', 'height': '45px', 'borderRadius': '12px', 'margin': '0 auto 15px',
                        'background': 'rgba(239, 68, 68, 0.2)',
                        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 
                        'color': THEME['danger'], 'fontWeight': 'bold', 'fontSize': '1.4rem'
                    }, children=['!']),
                    create_header_with_info("Critical Anomalies", f"Points above {86}% probability", align='center'),
                    html.H3(f"{critical_count:,}", style={'color': THEME['danger'], 'margin': 0, 'textAlign': 'center', 'fontSize': '1.8rem', 'fontWeight': '800'}),
                    html.P(f"{critical_pct:.1f}% of data", style={'textAlign': 'center', 'color': THEME['subtext'], 'fontSize': '0.8rem', 'marginTop': '5px'})
                ]),

                html.Div(style={
                    **card_style,
                    'borderLeft': f'5px solid {THEME["warning"]}',
                    'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, rgba(245, 158, 11, 0.05) 100%)'
                }, children=[
                    html.Div(style={
                        'width': '45px', 'height': '45px', 'borderRadius': '12px', 'margin': '0 auto 15px',
                        'background': 'rgba(245, 158, 11, 0.2)',
                        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 
                        'color': THEME['warning'], 'fontWeight': 'bold', 'fontSize': '1.4rem'
                    }, children=['~']),
                    create_header_with_info("Avg Probability", "Mean failure risk across dataset", align='center'),
                    html.H3(f"{avg_prob:.1f}%", style={'color': THEME['warning'], 'margin': 0, 'textAlign': 'center', 'fontSize': '1.8rem', 'fontWeight': '800'})
                ]),

                html.Div(style={
                    **card_style,
                    'borderLeft': f'5px solid {THEME["success"]}', 
                    'background': f'linear-gradient(135deg, {THEME["card_bg"]} 0%, rgba(16, 185, 129, 0.05) 100%)'
                }, children=[
                    html.Div(style={
                        'width': '45px', 'height': '45px', 'borderRadius': '12px', 'margin': '0 auto 15px',
                        'background': 'rgba(16, 185, 129, 0.2)',
                        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                        'color': THEME['success'], 'fontWeight': 'bold', 'fontSize': '1.4rem'
                    }, children=['↑']),
                    create_header_with_info("Peak Risk", "Highest recorded failure probability", align='center'),
                    html.H3(f"{max_prob:.1f}%", style={'color': THEME['success'], 'margin': 0, 'textAlign': 'center', 'fontSize': '1.8rem', 'fontWeight': '800'})
                ]),
            ]),

            html.Div(style={
                **card_style,
                'padding': '30px'
            }, children=[
                 create_header_with_info("Time-Series Analysis", "Visualize how failure probability evolves over time. Peaks above the red dashed line indicate critical failure risks.", align='left'),
                dcc.Graph(figure=fig, config={'scrollZoom': True, 'displayModeBar': True})
            ])
        ])
        
        return results, msg
        
    except Exception as e:
        error_card = html.Div(style={
            **card_style,
            'borderLeft': f'5px solid {THEME["danger"]}',
            'textAlign': 'center',
            'padding': '40px'
        }, children=[
            html.Div(style={
                'width': '80px',
                'height': '80px',
                'margin': '0 auto 20px',
                'borderRadius': '50%',
                'background': f'rgba(239, 68, 68, 0.2)',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'fontSize': '2.5rem',
                'color': THEME['danger'],
                'fontWeight': 'bold'
            }, children=['X']),
            html.H4("Upload Error", style={'color': THEME['danger'], 'marginBottom': '10px'}),
            html.P(f"{str(e)}", style={'color': THEME['subtext'], 'fontSize': '0.9rem'}),
            html.P("Please ensure your CSV file contains the required sensor columns.", style={'color': THEME['text'], 'fontSize': '0.85rem', 'marginTop': '15px'})
        ])
        return error_card, "ERROR: Processing failed"

if __name__ == '__main__':
    app.run(debug=True)