import pandas as pd
import numpy as np

def analyze_impressions():
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv('BDM_25T2_ROE_24ds3000044@ds.study.iitm.ac.in - dataset.csv')
    
    print(f"Dataset loaded successfully!")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Calculate total impressions for each Geo by summing all channel impressions
    impression_columns = ['Channel0_impression', 'Channel1_impression', 'Channel2_impression', 'Channel3_impression', 'Channel4_impression']
    
    print("Calculating total impressions by geography...")
    
    # Group by Geo and sum impressions
    geo_impressions = df.groupby('geo')[impression_columns].sum()
    
    # Calculate total impressions per Geo
    geo_impressions['Total_Impressions'] = geo_impressions.sum(axis=1)
    
    # Sort by total impressions to find the highest
    geo_impressions_sorted = geo_impressions.sort_values('Total_Impressions', ascending=False)
    
    print('Total Impressions by Geography:')
    print('=' * 60)
    for geo, row in geo_impressions_sorted.iterrows():
        print(f'{geo}: {row["Total_Impressions"]:,.0f}')
    
    print('\n' + '=' * 60)
    print(f'Geography with HIGHEST total impressions: {geo_impressions_sorted.index[0]}')
    print(f'Total Impressions: {geo_impressions_sorted.iloc[0]["Total_Impressions"]:,.0f}')
    
    # Show breakdown for the top Geo
    top_geo = geo_impressions_sorted.index[0]
    print(f'\nBreakdown for {top_geo}:')
    for col in impression_columns:
        print(f'  {col}: {geo_impressions_sorted.loc[top_geo, col]:,.0f}')
    
    # Show some statistics
    print(f'\nStatistics:')
    print(f'  Average total impressions per Geo: {geo_impressions["Total_Impressions"].mean():,.0f}')
    print(f'  Median total impressions per Geo: {geo_impressions["Total_Impressions"].median():,.0f}')
    print(f'  Standard deviation: {geo_impressions["Total_Impressions"].std():,.0f}')
    
    return geo_impressions_sorted

if __name__ == "__main__":
    try:
        results = analyze_impressions()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
