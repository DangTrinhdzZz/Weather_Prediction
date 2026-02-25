from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, List, Tuple, Optional, Sequence
from data_preprocessing import basic_preprocessing_hourly
import pandas as pd
# ------------------------------------
# CONFIG: File Input / Output
# ------------------------------------
INPUT = "data/raw data/hanoi_weather_data_hourly.csv"
OUTPUT = "data/hourly_to_daily_weather.csv"

def load_data(input_file):
    print(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    print(f"Loaded {len(df)} hourly records")
    return df

def process_conditions_6h(df):
    """
    Split by 6-hour windows and assign weather mode for each block.
    Output columns: cond_6h_0, cond_6h_1, ...
    """
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    # Gán block 6 giờ
    df["block"] = df["hour"] // 6  # 0,1,2,3

    # Lấy mode theo mỗi block
    cond_df = df.pivot_table(
        index="date",
        columns="block",
        values="conditions",
        aggfunc=lambda x: pd.Series.mode(x)[0] if len(x.mode()) > 0 else None
    )

    cond_df = cond_df.add_prefix("cond_6h_")
    return cond_df.reset_index()
def aggregate_daily(df):
    print(" Aggregating hourly → daily...")

    daily = df.groupby(['name', 'date']).agg(
        temp=('temp', 'mean'),
        tempmax=('temp', 'max'),
        tempmin=('temp', 'min'),
        feelslikemax=('feelslike', 'max'),
        feelslikemin=('feelslike', 'min'),
        feelslike=('feelslike', 'mean'),
        dew=('dew', 'mean'),
        humidity=('humidity', 'mean'),
        precip=('precip', 'sum'),
        windgust=('windgust', 'mean'),
        windspeed=('windspeed', 'mean'),
        winddir=('winddir', 'mean'),
        sealevelpressure=('sealevelpressure', 'mean'),
        cloudcover=('cloudcover', 'mean'),
        visibility=('visibility', 'mean'),
        solarradiation=('solarradiation', 'mean'),
        solarenergy=('solarenergy', 'sum'),
        uvindex=('uvindex', 'mean')
    ).reset_index()
    
    daily.columns = [
        'name', 'datetime','temp',
        'tempmax','tempmin',
        'feelslikemax','feelslikemin','feelslike',
        'dew','humidity','precip',
        'windgust','windspeed','winddir',
        'sealevelpressure','cloudcover','visibility',
        'solarradiation','solarenergy','uvindex'
    ]

    print("Done converting to daily format!")
    return daily

def additional_hourly_feature(df):
    """
    Create additional 6-hourly aggregated features from hourly data.
    - Temperature-related variables (temp, feelslike, dew) are aggregated by mean, max, and min
      for 4 time windows per day:
        * 0h–6h  → 6h_next0
        * 6h–12h → 6h_next1
        * 12h–18h → 6h_next2
        * 18h–24h → 6h_next3
    - Sun-related variables (solarradiation, solarenergy, uvindex) are aggregated by daily max.
    """

    print("Aggregating hourly → 6-hourly...")

    # Ensure datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date

    # Define 6-hour window groups
    def hour_bin(h):
        if 0 <= h < 6:
            return '6h_next0'
        elif 6 <= h < 12:
            return '6h_next1'
        elif 12 <= h < 18:
            return '6h_next2'
        else:
            return '6h_next3'

    df['hour_bin'] = df['hour'].apply(hour_bin)

    # Temperature-related variables
    temp_agg = df.groupby(['name', 'date', 'hour_bin']).agg(
        temp_mean=('temp', 'mean'),
        temp_max=('temp', 'max'),
        temp_min=('temp', 'min'),
        feelslike_mean=('feelslike', 'mean'),
        feelslike_max=('feelslike', 'max'),
        feelslike_min=('feelslike', 'min'),
        dew_mean=('dew', 'mean'),
        dew_max=('dew', 'max'),
        dew_min=('dew', 'min')
    ).reset_index()

    # Pivot 6-hour bins to columns
    temp_pivot = temp_agg.pivot(index=['name', 'date'], columns='hour_bin')
    temp_pivot.columns = [f"{col[0]}_{col[1]}" for col in temp_pivot.columns]
    temp_pivot = temp_pivot.reset_index()

    # Sun-related daily max
    sun_agg = df.groupby(['name', 'date']).agg(
        solarradiation_max=('solarradiation', 'max'),
        solarenergy_max=('solarenergy', 'max'),
        uvindex_max=('uvindex', 'max')
    ).reset_index()

    # Merge
    result = pd.merge(temp_pivot, sun_agg, on=['name', 'date'], how='left')

    return result

if __name__ == "__main__":
    print("Running weather data processing pipeline…")

    df = load_data(INPUT)
    daily_df = aggregate_daily(df)
    cond_df = process_conditions_6h(df)
    daily_additional_df = additional_hourly_feature(df)
    merged_df = (
        daily_df.merge(
            cond_df,
            left_on='datetime',
            right_on='date',
            how='left'
        )
        .merge(
            daily_additional_df,
            left_on='datetime',
            right_on='date',
            how='left'
        )
    )

    # Giữ lại cột datetime duy nhất
    merged_df = merged_df.drop(columns=['date_x', 'date_y', 'name_x', 'name_y'], errors='ignore')
    print("\nSample output (first 5 rows):")
    print(merged_df.columns)
    

    merged_df.to_csv(OUTPUT, index=False)
    print(f"\n✅ Output saved to: {OUTPUT}")