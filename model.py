import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#ffffff'

# AQI Categories
AQI_CATEGORIES = {
    'Good': {'range': (0, 50), 'color': '#2ca02c', 'advice': 'Air quality is satisfactory. Enjoy outdoor activities!'},
    'Satisfactory': {'range': (51, 100), 'color': '#90ee90', 'advice': 'Air quality is acceptable. Sensitive groups should limit outdoor exposure.'},
    'Moderately Polluted': {'range': (101, 200), 'color': '#ffd700', 'advice': 'Members of sensitive groups should reduce outdoor activities. Wear N95 masks if going out.'},
    'Poor': {'range': (201, 300), 'color': '#ff8c00', 'advice': 'Avoid outdoor activities. Wear N95/N99 masks. Use air purifiers indoors.'},
    'Very Poor': {'range': (301, 400), 'color': '#ff4500', 'advice': 'Exposure may cause respiratory illness. Stay indoors. Use air purifiers. Consult doctor.'},
    'Severe': {'range': (401, 500), 'color': '#8b0000', 'advice': 'Serious health effects. Remain indoors. Wear N95/N99 masks. Minimize outdoor exposure.'},
    'Hazardous': {'range': (501, 600), 'color': '#000000', 'advice': 'EMERGENCY! Avoid all outdoor activities. Wear protective masks. Stay in air-purified rooms.'}
}

def get_aqi_category(aqi_value):
    """Returns category and details for given AQI value"""
    for category, details in AQI_CATEGORIES.items():
        if details['range'][0] <= aqi_value <= details['range'][1]:
            return category, details
    return 'Hazardous', AQI_CATEGORIES['Hazardous']

# Load data
df = pd.read_csv('city_day.csv')

# Display cities
print("\n" + "="*70)
print("AVAILABLE CITIES")
print("="*70)
cities = sorted(df['City'].unique())
for i, city in enumerate(cities, 1):
    print(f"  {i}. {city}")

# Get city choice
while True:
    try:
        choice = int(input(f"\nEnter city number (1-{len(cities)}): "))
        if 1 <= choice <= len(cities):
            city_name = cities[choice - 1]
            break
        print(f"Please enter a valid number between 1 and {len(cities)}")
    except ValueError:
        print("Invalid input!")

# Filter and prepare data
city_data = df[df['City'] == city_name].copy()
city_data['Date'] = pd.to_datetime(city_data['Date'])
city_data = city_data.sort_values('Date').reset_index(drop=True)

# Clean data
city_data['AQI'] = city_data['AQI'].fillna(method='ffill').fillna(method='bfill')
city_data['AQI'] = city_data['AQI'].interpolate(method='linear')
city_data = city_data.dropna(subset=['AQI']).reset_index(drop=True)

# Remove outliers beacuse it makes the model skewed
Q1 = city_data['AQI'].quantile(0.25)
Q3 = city_data['AQI'].quantile(0.75)
IQR = Q3 - Q1
city_data['AQI'] = city_data['AQI'].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

# Prepare for Prophet
prophet_data = city_data[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})

# Train Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(prophet_data)

# Calculate RMSE
predictions = model.predict(prophet_data[['ds']])
rmse = np.sqrt(mean_squared_error(prophet_data['y'].values, predictions['yhat'].values))

# Make 10-day forecast
future_dates = model.make_future_dataframe(periods=10)
forecast = model.predict(future_dates)
forecast_10days = forecast[forecast['ds'] > prophet_data['ds'].max()][['ds', 'yhat']].reset_index(drop=True)

# Clip values
min_aqi = prophet_data['y'].min()
max_aqi = prophet_data['y'].max()
forecast_10days['yhat'] = forecast_10days['yhat'].clip(lower=min_aqi * 0.5, upper=max_aqi * 1.2)

# Calculate average
avg_aqi = forecast_10days['yhat'].mean()
avg_category, avg_details = get_aqi_category(avg_aqi)

# Print results
print("\n" + "="*70)
print(f"AQI FORECAST FOR {city_name.upper()}")
print("="*70)
print(f"Model RMSE: {rmse:.2f}\n")
print("10-Day Predictions:")
print("-" * 70)

for i, row in forecast_10days.iterrows():
    aqi_val = row['yhat']
    category, _ = get_aqi_category(aqi_val)
    print(f"Day {i+1} ({row['ds'].strftime('%Y-%m-%d')}): {aqi_val:.2f} AQI - {category.upper()}")

print("-" * 70)
print(f"\nAverage AQI (10 Days): {avg_aqi:.2f}")
print(f"Category: {avg_category.upper()}")
print(f"Health Advice: {avg_details['advice']}")
print("="*70)

# Create plot
fig, ax = plt.subplots(figsize=(14, 7))

# Prepare plot data
days = [f'Day {i+1}' for i in range(len(forecast_10days))]
aqi_values = forecast_10days['yhat'].values

# Plot line
ax.plot(days, aqi_values, color='#2c3e50', linewidth=3, marker='o', markersize=4, 
        markerfacecolor='#3498db', markeredgecolor='white', markeredgewidth=1, label='AQI Trend')

# Add value labels
for i, (day, aqi) in enumerate(zip(days, aqi_values)):
    ax.text(i, aqi + 15, f'{aqi:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Labels and title
ax.set_xlabel('Days', fontsize=12, fontweight='bold', color='#2c3e50')
ax.set_ylabel('AQI Level', fontsize=12, fontweight='bold', color='#2c3e50')
ax.set_title(f'10-Day AQI Forecast - {city_name}', fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
ax.set_ylim(0, 550)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper left', frameon=True)

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors='#2c3e50', labelsize=10)

# Add category legend
legend_items = [
    ('Good', '0-50', '#2ca02c'),
    ('Satisfactory', '51-100', '#90ee90'),
    ('Polluted', '101-300', '#ffd700'),
    ('Poor', '301-400', '#ff4500'),
    ('Hazardous', '401-600', '#8b0000')
]
 
y_pos = 0.95
for category, aqi_range, color in legend_items:
    ax.text(1.01, y_pos, '█', transform=ax.transAxes, fontsize=16, color=color, va='top')
    ax.text(1.07, y_pos, f"{category}\n{aqi_range}", transform=ax.transAxes, fontsize=8, 
            va='top', fontweight='bold', color='#2c3e50')
    y_pos -= 0.15

plt.tight_layout()
plt.subplots_adjust(right=0.78)
plt.show()
